"""Run a frozen 100-query stopping comparison with durable round checkpoints."""

import ast
import fcntl
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import experiment_common as p
from full100_core import ApiFailure, DurableAPI, atomic_json

OUT = p.ROOT
LLM_URL = p.BASE + "/chat/completions"
JEV_URL = "https://api.typesafe.ai/v1/systemone"
SEED = 20260922
COUNTS = {"compositional": 41, "comparison": 24, "bridge_comparison": 24, "inference": 11}
MAX_ROUNDS = 7
CONCURRENCY = 8


def make_manifest(data):
    rng = random.Random(SEED)
    selected = []
    for kind, count in COUNTS.items():
        candidates = sorted(
            [i for i, row in enumerate(data) if row["type"] == kind], key=lambda i: data[i]["_id"]
        )
        selected.extend(rng.sample(candidates, count))
    selected.sort()
    return {
        "seed": SEED,
        "indices": selected,
        "ids": [data[i]["_id"] for i in selected],
        "type_counts": COUNTS,
        "max_rounds": MAX_ROUNDS,
        "concurrency": CONCURRENCY,
        "llm_interval_seconds": 2.3,
        "model": p.MODEL,
        "base_url": p.BASE,
        "jev_model": "jev-1.13.0",
        "jev_threshold": 0.5,
        "jev_question": p.QUESTION,
        "source_sha256": hashlib.sha256(p.source.encode()).hexdigest(),
        "dataset_sha256": hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(),
        "embedding_manifest": json.loads((p.ROOT / "embedding_manifest.json").read_text()),
        "protocol": "One full trajectory per query. Same main question and accumulated Q&A for both stopping models. First affirmative decision, maximum seven rounds. No final-answer quality scoring. No prompt or threshold tuning on this set.",
    }


def clean_text(raw):
    choice = raw["choices"][0]
    if choice.get("finish_reason") != "stop":
        raise ValueError("Incomplete LLM output")
    text = choice["message"]["content"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Empty LLM output")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    if not text:
        raise ValueError("Empty visible LLM output")
    return text


def parse_indices(text, count):
    cleaned = re.sub(r"^```(?:python|json)?\s*|\s*```$", "", text.strip())
    try:
        indices = ast.literal_eval(cleaned)
    except (SyntaxError, ValueError) as exc:
        raise ValueError("Invalid supporting document syntax") from exc
    if not isinstance(indices, list) or any(
        type(i) is not int or not 0 <= i < count for i in indices
    ):
        raise ValueError("Invalid supporting document indices")
    return indices


class Experiment:
    def __init__(self, api, client, worker):
        self.api, self.client, self.worker = api, client, worker
        self.embedding_lock = threading.Lock()
        self.search_lock = threading.Lock()
        self.key_lock = threading.Lock()
        self.abort = threading.Event()
        self.exhausted = set()

    def llm(self, prompt, label, validator=None):
        if self.abort.is_set():
            raise ApiFailure("run", "aborted")

        def validate(raw):
            text = clean_text(raw)
            if validator:
                validator(text)

        body = {
            "model": p.MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 2048,
            "thinking": {"type": "disabled"},
        }
        try:
            raw = self.api.request("llm", LLM_URL, os.environ["LLM_API_KEY"], body, label, validate)
        except ApiFailure as exc:
            if exc.status in [401, 402, 403]:
                self.abort.set()
            raise
        return clean_text(raw)

    def jev(self, query, contexts, label):
        body = {
            "model": "jev-1.13.0",
            "state": {"main_query": query, "intermediate_context": "\n".join(contexts)},
            "questions": {"enough_information": p.QUESTION},
        }

        def validate(raw):
            score = raw["answers"]["enough_information"]["noul"]
            if type(score) not in (int, float) or not 0 <= score <= 1:
                raise ValueError("Invalid Jev score")

        with self.key_lock:
            for name in [
                name.strip()
                for name in os.environ.get("JEV_KEY_NAMES", "TYPESAFE_API_KEY").split(",")
            ]:
                if name in self.exhausted or not os.getenv(name):
                    continue
                try:
                    raw = self.api.request("jev", JEV_URL, os.environ[name], body, label, validate)
                    return raw["answers"]["enough_information"]["noul"]
                except ApiFailure as exc:
                    if exc.status in [401, 402, 403]:
                        self.exhausted.add(name)
                        atomic_json(OUT / "unavailable_credentials.json", sorted(self.exhausted))
                        continue
                    raise
            self.abort.set()
            raise ApiFailure("jev", "no_available_credential")

    def search(self, text):
        digest = hashlib.sha256(text.encode()).hexdigest()
        cache = OUT / "retrieval-cache" / f"{digest}.json"
        with self.embedding_lock:
            if cache.exists():
                return json.loads(cache.read_text())["documents"]
            self.worker.stdin.write(json.dumps({"query": text}) + "\n")
            self.worker.stdin.flush()
            vector = json.loads(self.worker.stdout.readline())["vector"]
        with self.search_lock:
            hits = self.client.search(
                "bge_pilot", data=[vector], limit=5, output_fields=["text", "reference", "metadata"]
            )[0]
        docs = p.unique([dict(h["entity"]) for h in hits])
        if not docs:
            raise ValueError("Empty retrieval result")
        atomic_json(cache, {"query": text, "documents": docs})
        return docs

    def run_query(self, item):
        qi, sample = item
        path = OUT / "traces" / f"{qi}.json"
        trace = (
            json.loads(path.read_text())
            if path.exists()
            else {
                "idx": qi,
                "id": sample["_id"],
                "type": sample["type"],
                "query": sample["question"],
                "gold_titles": sorted({x[0] for x in sample["supporting_facts"]}),
                "rounds": [],
            }
        )
        assert trace["id"] == sample["_id"]
        query, rounds = trace["query"], trace["rounds"]
        gold = set(trace["gold_titles"])
        contexts = [
            f"Intermediate query{r['round']}: {r['subquery']}\nIntermediate answer{r['round']}: {r['answer']}"
            for r in rounds
        ]
        cumulative = p.unique([d for r in rounds for d in r["selected"]])
        for ri in range(len(rounds), MAX_ROUNDS):
            if self.abort.is_set():
                raise ApiFailure("run", "aborted")
            label = f"q{qi}-r{ri + 1}"
            subquery = self.llm(
                p.PROMPTS["FOLLOWUP_QUERY_PROMPT"].format(
                    query=query, intermediate_context="\n".join(contexts)
                ),
                label + "-subquery",
            )
            docs = self.search(subquery)
            answer = self.llm(
                p.PROMPTS["INTERMEDIATE_ANSWER_PROMPT"].format(
                    retrieved_documents=p.format_docs(docs), sub_query=subquery
                ),
                label + "-answer",
            )
            selected = []
            if "No relevant information found" not in answer:
                selection = self.llm(
                    p.PROMPTS["GET_SUPPORTED_DOCS_PROMPT"].format(
                        retrieved_documents=p.format_docs(docs), query=subquery, answer=answer
                    ),
                    label + "-selection",
                    lambda text: parse_indices(text, len(docs)),
                )
                selected = [docs[i] for i in parse_indices(selection, len(docs))]
            cumulative = p.unique(cumulative + selected)
            contexts.append(
                f"Intermediate query{ri + 1}: {subquery}\nIntermediate answer{ri + 1}: {answer}"
            )

            def validate_stop(text):
                if text.lower() not in ["yes", "no"]:
                    raise ValueError("Invalid stop output")

            stop = self.llm(
                p.PROMPTS["REFLECTION_PROMPT"].format(
                    query=query, intermediate_context="\n".join(contexts)
                ),
                label + "-stop",
                validate_stop,
            )
            score = self.jev(query, contexts, label + "-jev")
            rounds.append(
                {
                    "round": ri + 1,
                    "subquery": subquery,
                    "answer": answer,
                    "retrieved": docs,
                    "selected": selected,
                    "llm_stop": stop.lower() == "yes",
                    "jev_score": score,
                    "metrics": p.metrics(cumulative, gold),
                }
            )
            atomic_json(path, trace)
            print(
                json.dumps(
                    {
                        "query": qi,
                        "round": ri + 1,
                        "coverage": rounds[-1]["metrics"]["coverage"],
                        "llm_stop": rounds[-1]["llm_stop"],
                        "jev_score": score,
                    }
                ),
                flush=True,
            )
        return qi


def progress(state, errors=None, **extra):
    traces = [json.loads(f.read_text()) for f in (OUT / "traces").glob("*.json")]
    atomic_json(
        OUT / "status.json",
        {
            "state": state,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "completed_queries": sum(len(t["rounds"]) == MAX_ROUNDS for t in traces),
            "recorded_rounds": sum(len(t["rounds"]) for t in traces),
            "target_queries": 100,
            "target_rounds": 700,
            "errors": errors or [],
            **extra,
        },
    )


def run():
    from pymilvus import MilvusClient

    OUT.mkdir(exist_ok=True)
    lock_file = (OUT / "runner.lock").open("w")
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for name in ["traces", "retrieval-cache"]:
        (OUT / name).mkdir(exist_ok=True)
    data = json.loads((p.SOURCE / "examples/data/2wikimultihopqa.json").read_text())
    manifest = make_manifest(data)
    if (OUT / "manifest.json").exists():
        assert manifest == json.loads((OUT / "manifest.json").read_text()), (
            "Manifest changed; refusing to mix experiments"
        )
    else:
        atomic_json(OUT / "manifest.json", manifest)
    if len(manifest["indices"]) != 100 or len(set(manifest["ids"])) != 100:
        raise ValueError("Invalid sample manifest")
    api = DurableAPI(OUT, p.CACHE)
    worker = None
    client = None
    errors = []
    heartbeat_stop = threading.Event()
    progress("starting", pid=os.getpid())

    def heartbeat():
        while not heartbeat_stop.wait(15):
            progress(
                "running",
                errors,
                pid=os.getpid(),
                active_requests=api.active,
                peak_active_requests=api.peak_active,
            )

    heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
    heartbeat_thread.start()
    start = time.monotonic()
    try:
        client = MilvusClient(str(p.ROOT / "milvus.db"))
        assert client.get_collection_stats("bge_pilot")["row_count"] == 6119
        worker = subprocess.Popen(
            [sys.executable, str(p.HERE / "bge_worker.py")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        assert json.loads(worker.stdout.readline())["ready"]
        experiment = Experiment(api, client, worker)
        for sweep in range(3):
            pending = [
                (i, data[i])
                for i in manifest["indices"]
                if not (OUT / "traces" / f"{i}.json").exists()
                or len(json.loads((OUT / "traces" / f"{i}.json").read_text())["rounds"])
                < MAX_ROUNDS
            ]
            if not pending or experiment.abort.is_set():
                break
            errors.clear()
            with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
                futures = {pool.submit(experiment.run_query, item): item[0] for item in pending}
                for future in as_completed(futures):
                    qi = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        failure = {
                            "idx": qi,
                            "sweep": sweep + 1,
                            "error": type(exc).__name__,
                            "reason": str(exc)[:100]
                            if isinstance(exc, (ApiFailure, ValueError))
                            else "See error type; no secret-bearing exception text logged",
                        }
                        errors.append(failure)
                        api.event({"kind": "query_failure", **failure})
                        print(json.dumps(failure), flush=True)
            if errors and not experiment.abort.is_set() and sweep < 2:
                time.sleep(60)
        complete = sum(
            len(json.loads(f.read_text())["rounds"]) == MAX_ROUNDS
            for f in (OUT / "traces").glob("*.json")
        )
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=30)
        progress(
            "ready_for_audit" if complete == 100 else "needs_attention",
            errors,
            wall_seconds_this_run=time.monotonic() - start,
            peak_active_requests=api.peak_active,
        )
        return complete == 100
    finally:
        heartbeat_stop.set()
        if worker:
            worker.stdin.close()
            try:
                worker.wait(timeout=30)
            except subprocess.TimeoutExpired:
                worker.terminate()
                worker.wait(timeout=15)
        if client:
            client.close()
        api.close()


if __name__ == "__main__":
    try:
        completed = run()
    except BlockingIOError:
        print("Another runner holds the experiment lock", flush=True)
        raise SystemExit(3)
    except Exception as exc:
        progress(
            "needs_attention",
            [{"error": type(exc).__name__, "reason": "Runner initialization or system failure"}],
        )
        print(json.dumps({"fatal": type(exc).__name__}), flush=True)
        raise SystemExit(1)
    raise SystemExit(0 if completed else 2)
