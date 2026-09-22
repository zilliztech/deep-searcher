"""Offline tests for rate control, retries, caching, and sample determinism."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from full100_core import DurableAPI, RateGate, retry_delay
from run_full100 import clean_text, make_manifest, parse_indices


class FakeGate:
    def __init__(self):
        self.waits = 0
        self.deferrals = []

    def wait(self):
        self.waits += 1

    def defer(self, seconds):
        self.deferrals.append(seconds)


class FakeHTTP:
    def __init__(self, values):
        self.values = list(values)
        self.calls = 0

    def post(self, *args, **kwargs):
        self.calls += 1
        value = self.values.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    def close(self):
        pass


def response(status=200, value=None, headers=None):
    return httpx.Response(
        status, json=value if value is not None else {"value": "ok"}, headers=headers or {}
    )


def validate(raw):
    if raw["value"] != "ok":
        raise ValueError("bad result")


class Tests(unittest.TestCase):
    def test_global_rate_spacing_and_cooldown(self):
        now = [0.0]
        gate = RateGate(2.3, clock=lambda: now[0], sleep=lambda d: now.__setitem__(0, now[0] + d))
        gate.wait()
        gate.wait()
        self.assertAlmostEqual(now[0], 2.3)
        gate.defer(60)
        gate.wait()
        self.assertAlmostEqual(now[0], 62.3)

    def test_retry_after(self):
        self.assertEqual(retry_delay("60", 0), 60)
        self.assertEqual(retry_delay(None, 1), 4)

    def test_429_backoff_and_durable_reuse(self):
        with tempfile.TemporaryDirectory() as d:
            http = FakeHTTP([response(429, headers={"Retry-After": "75"}), response()])
            api = DurableAPI(d, http=http, sleep=lambda _: None)
            gate = FakeGate()
            api.gates["llm"] = gate
            api.request("llm", "https://test.invalid", "secret-test-key", {}, "first", validate)
            self.assertEqual(gate.deferrals, [75])
            restored = DurableAPI(d, http=FakeHTTP([]))
            raw = restored.request(
                "llm", "https://test.invalid", "another-key", {}, "second", validate
            )
            self.assertEqual(raw["value"], "ok")
            self.assertEqual(restored.http.calls, 0)
            self.assertNotIn("secret-test-key", (Path(d) / "attempts.jsonl").read_text())

    def test_invalid_output_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            api = DurableAPI(
                d,
                http=FakeHTTP([response(value={"value": "bad"}), response()]),
                sleep=lambda _: None,
            )
            api.gates["llm"] = FakeGate()
            self.assertEqual(
                api.request("llm", "https://test.invalid", "key", {}, "request", validate)["value"],
                "ok",
            )
            self.assertEqual(api.http.calls, 2)
            self.assertEqual(len(list((Path(d) / "failed-responses").glob("*.json"))), 1)

    def test_transport_failure_retries(self):
        with tempfile.TemporaryDirectory() as d:
            api = DurableAPI(
                d, http=FakeHTTP([httpx.ReadTimeout("timeout"), response()]), sleep=lambda _: None
            )
            api.gates["llm"] = FakeGate()
            api.request("llm", "https://test.invalid", "key", {}, "request", validate)
            self.assertEqual(api.http.calls, 2)

    def test_parser_rejects_incomplete_and_invalid_indices(self):
        self.assertEqual(parse_indices("```python\n[0, 2]\n```", 3), [0, 2])
        for value in ["[-1]", "[3]", "[True]", "[0,"]:
            with self.assertRaises(ValueError):
                parse_indices(value, 3)
        with self.assertRaises(ValueError):
            clean_text({"choices": [{"finish_reason": "length", "message": {"content": "Yes"}}]})

    def test_concurrent_identical_requests_share_cache(self):
        from concurrent.futures import ThreadPoolExecutor

        with tempfile.TemporaryDirectory() as d:
            api = DurableAPI(d, http=FakeHTTP([response()]), sleep=lambda _: None)
            api.gates["llm"] = FakeGate()
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(
                        api.request, "llm", "https://test.invalid", "key", {}, name, validate
                    )
                    for name in ["one", "two"]
                ]
                self.assertTrue(all(f.result()["value"] == "ok" for f in futures))
            self.assertEqual(api.http.calls, 1)

    def test_round_checkpoint_resume(self):
        import run_full100 as experiment_module
        from full100_core import ApiFailure

        with tempfile.TemporaryDirectory() as d, patch.object(experiment_module, "OUT", Path(d)):
            experiment = experiment_module.Experiment(None, None, None)
            labels = []

            def llm(prompt, label, validator=None):
                labels.append(label)
                text = (
                    "[0]"
                    if label.endswith("-selection")
                    else "Yes"
                    if label.endswith("-stop")
                    else "Evidence"
                )
                if validator:
                    validator(text)
                return text

            experiment.llm = llm
            experiment.search = lambda text: [
                {"text": "Evidence", "metadata": {"title": "Gold"}, "reference": "doc"}
            ]

            def interrupted(query, contexts, label):
                if "-r3-" in label:
                    raise ApiFailure("jev", "transport_error")
                return 0.9

            experiment.jev = interrupted
            sample = {
                "_id": "sample",
                "type": "inference",
                "question": "Question",
                "supporting_facts": [["Gold", 0]],
            }
            with self.assertRaises(ApiFailure):
                experiment.run_query((0, sample))
            checkpoint = json.loads((Path(d) / "traces/0.json").read_text())
            self.assertEqual(len(checkpoint["rounds"]), 2)
            labels.clear()
            experiment.jev = lambda *args: 0.9
            experiment.run_query((0, sample))
            self.assertTrue(all("-r1-" not in label and "-r2-" not in label for label in labels))
            self.assertEqual(len(json.loads((Path(d) / "traces/0.json").read_text())["rounds"]), 7)

    def test_sampling_is_frozen_and_stratified(self):
        data = json.loads(
            (
                __import__("experiment_common").SOURCE / "examples/data/2wikimultihopqa.json"
            ).read_text()
        )
        with patch.object(
            __import__("experiment_common"), "ROOT", Path(__file__).parent / "fixtures"
        ):
            a, b = make_manifest(data), make_manifest(data)
        self.assertEqual(a, b)
        frozen = json.loads((Path(__file__).parent / "results/manifest.json").read_text())
        self.assertEqual(a["ids"], frozen["ids"])
        self.assertEqual(a["indices"], frozen["indices"])
        self.assertEqual(len(set(a["ids"])), 100)
        from collections import Counter

        self.assertEqual(dict(Counter(data[i]["type"] for i in a["indices"])), a["type_counts"])


if __name__ == "__main__":
    unittest.main()
