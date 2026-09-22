"""Rate-limited, durable API execution for the stopping experiment."""

import hashlib
import json
import os
import random
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import httpx


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{threading.get_ident()}.tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def retry_delay(header, attempt):
    if header:
        try:
            return max(0.0, float(header))
        except ValueError:
            try:
                return max(
                    0.0,
                    (parsedate_to_datetime(header) - datetime.now(timezone.utc)).total_seconds(),
                )
            except (ValueError, TypeError):
                pass
    return min(60.0, 2.0 ** min(attempt + 1, 6))


class RateGate:
    def __init__(self, interval=2.3, clock=time.monotonic, sleep=time.sleep):
        self.interval = interval
        self.clock = clock
        self.sleep = sleep
        self.next_start = 0.0
        self.cooldown = 0.0
        self.lock = threading.Lock()

    def wait(self):
        while True:
            with self.lock:
                now = self.clock()
                delay = max(self.next_start, self.cooldown) - now
                if delay <= 0:
                    self.next_start = now + self.interval
                    return
            self.sleep(min(delay, 1.0))

    def defer(self, seconds):
        with self.lock:
            self.cooldown = max(self.cooldown, self.clock() + seconds)


class ApiFailure(RuntimeError):
    def __init__(self, kind, status):
        self.kind, self.status = kind, status
        super().__init__(f"{kind}: {status}")


class DurableAPI:
    def __init__(self, root, legacy_cache=None, http=None, sleep=time.sleep, max_attempts=8):
        self.root = Path(root)
        self.cache = self.root / "cache"
        self.legacy_cache = Path(legacy_cache) if legacy_cache else None
        self.http = http or httpx.Client(
            timeout=httpx.Timeout(120, connect=20),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=20),
        )
        self.sleep = sleep
        self.max_attempts = max_attempts
        self.gates = {"llm": RateGate(), "jev": RateGate(interval=0.1)}
        self.lock = threading.RLock()
        self.cache_locks = {}
        self.exhausted_keys = set()
        self.active = 0
        self.peak_active = 0
        self.started = time.time()
        for name in ["cache", "calls", "failed-responses"]:
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def event(self, value):
        value = {"timestamp": datetime.now(timezone.utc).isoformat(), **value}
        with self.lock:
            with (self.root / "attempts.jsonl").open("a") as handle:
                handle.write(json.dumps(value) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    def request(self, kind, url, key, body, label, validate):
        digest = hashlib.sha256(
            json.dumps({"url": url, "body": body}, sort_keys=True).encode()
        ).hexdigest()
        with self.lock:
            guard = self.cache_locks.setdefault(digest, threading.Lock())
        with guard:
            for candidate in [self.cache / (digest + ".json")] + (
                [self.legacy_cache / (digest + ".json")] if self.legacy_cache else []
            ):
                if candidate.exists():
                    try:
                        record = json.loads(candidate.read_text())
                        validate(record["response"])
                    except (ValueError, KeyError, TypeError, AssertionError):
                        continue
                    if candidate.parent != self.cache:
                        atomic_json(self.cache / candidate.name, record)
                    self.record_call(label, digest, record, True)
                    return record["response"]
            for attempt in range(self.max_attempts):
                self.gates[kind].wait()
                start = time.monotonic()
                with self.lock:
                    self.active += 1
                    self.peak_active = max(self.peak_active, self.active)
                status, raw, headers = "transport_error", None, {}
                try:
                    response = self.http.post(
                        url, headers={"Authorization": "Bearer " + key}, json=body
                    )
                    status, headers = response.status_code, response.headers
                    if status == 200:
                        try:
                            raw = response.json()
                            validate(raw)
                        except (ValueError, KeyError, TypeError, AssertionError):
                            status = "invalid_output"
                except httpx.TransportError:
                    status = "transport_error"
                finally:
                    seconds = time.monotonic() - start
                    with self.lock:
                        self.active -= 1
                event = {
                    "kind": kind,
                    "label": label,
                    "digest": digest,
                    "attempt": attempt + 1,
                    "status": status,
                    "seconds": seconds,
                }
                if isinstance(raw, dict):
                    event["usage"] = raw.get("usage", {})
                self.event(event)
                if status == 200:
                    record = {"kind": kind, "request": body, "response": raw, "seconds": seconds}
                    atomic_json(self.cache / (digest + ".json"), record)
                    self.record_call(label, digest, record, False)
                    return raw
                if raw is not None:
                    atomic_json(
                        self.root / "failed-responses" / f"{digest}-{time.time_ns()}.json",
                        {"request": body, "response": raw},
                    )
                retriable = status in [
                    408,
                    429,
                    500,
                    502,
                    503,
                    504,
                    "transport_error",
                    "invalid_output",
                ]
                if not retriable or attempt + 1 == self.max_attempts:
                    raise ApiFailure(kind, status)
                if status == "invalid_output" and attempt >= 2:
                    raise ApiFailure(kind, status)
                delay = retry_delay(headers.get("Retry-After"), attempt)
                if status == 429:
                    self.gates[kind].defer(max(delay, 60.0))
                else:
                    self.sleep(delay + random.random())
            raise ApiFailure(kind, "attempts_exhausted")

    def record_call(self, label, digest, record, cached):
        atomic_json(
            self.root / "calls" / (label + ".json"),
            {
                "label": label,
                "digest": digest,
                "kind": record["kind"],
                "cached_on_latest_execution": cached,
                "seconds": record["seconds"],
                "usage": record["response"].get("usage", {}),
            },
        )

    def close(self):
        self.http.close()
