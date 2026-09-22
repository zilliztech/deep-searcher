"""Verify the published aggregates offline from per-round evidence and usage."""

import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import experiment_common as p

SUMMARY = Path(__file__).resolve().parent / "results"
ROOT = Path(__file__).resolve().parent / "artifacts" / "results"


def main():
    if not (ROOT / "calls.json").exists():
        raise SystemExit("Run uv run python fetch_artifacts.py before replaying results.")
    report = json.loads((SUMMARY / "report.json").read_text())
    manifest = json.loads((SUMMARY / "manifest.json").read_text())
    calls = json.loads((ROOT / "calls.json").read_text())
    expected_rows = json.loads((ROOT / "results.json").read_text())
    data = json.loads((p.SOURCE / "examples/data/2wikimultihopqa.json").read_text())
    assert (
        hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        == manifest["dataset_sha256"]
    )
    assert hashlib.sha256(p.source.encode()).hexdigest() == manifest["source_sha256"]
    summaries = {name: [] for name in report["policies"]}
    behavior = Counter()
    agreement = []
    usage = {name: Counter() for name in summaries}
    for row in expected_rows:
        trace = json.loads((ROOT / "traces" / f"{row['idx']}.json").read_text())
        assert trace["id"] == row["id"] == data[row["idx"]]["_id"]
        assert len(trace["rounds"]) == 7
        docs = []
        for i, state in enumerate(trace["rounds"], 1):
            assert state["round"] == i
            assert all(d in state["retrieved"] for d in state["selected"])
            docs = p.unique(docs + state["selected"])
            assert p.metrics(docs, set(trace["gold_titles"])) == state["metrics"]
            agreement.append(state["llm_stop"] == (state["jev_score"] >= 0.5))
        for name in summaries:
            if name == "llm":
                chosen = next((s for s in trace["rounds"] if s["llm_stop"]), None)
            elif name == "jev":
                chosen = next((s for s in trace["rounds"] if s["jev_score"] >= 0.5), None)
            else:
                chosen = None
            chosen = chosen or trace["rounds"][-1]
            assert row["policies"][name]["round"] == chosen["round"]
            for metric in ["r2", "r5", "coverage", "all_supports"]:
                assert row["policies"][name][metric] == chosen["metrics"][metric]
            summaries[name].append({"round": chosen["round"], **chosen["metrics"]})
            for ri in range(1, chosen["round"] + 1):
                for stage in ["subquery", "answer", "selection"] + (
                    ["stop"] if name == "llm" else ["jev"] if name == "jev" else []
                ):
                    call = calls.get(f"q{row['idx']}-r{ri}-{stage}")
                    if call:
                        kind = call["kind"]
                        usage[name][kind + "_requests"] += 1
                        for key, dest in [
                            ("prompt_tokens", "llm_input_tokens"),
                            ("completion_tokens", "llm_output_tokens"),
                            ("input_tokens", "jev_input_tokens"),
                        ]:
                            usage[name][dest] += call["usage"].get(key, 0)
        a, b = row["policies"]["llm"]["round"], row["policies"]["jev"]["round"]
        behavior["same" if a == b else "jev_earlier" if b < a else "jev_later"] += 1
    assert len(expected_rows) == 100 and len(agreement) == 700
    assert [r["idx"] for r in expected_rows] == manifest["indices"]
    assert dict(Counter(r["type"] for r in expected_rows)) == manifest["type_counts"]
    for name, rows in summaries.items():
        for metric in ["round", "r2", "r5", "coverage", "all_supports"]:
            assert math.isclose(
                statistics.mean(r[metric] for r in rows), report["policies"][name][metric]
            )
        for metric, value in usage[name].items():
            assert value == report["policies"][name][metric], (name, metric)
    assert dict(behavior) == report["first_stop_behavior"]
    assert statistics.mean(agreement) == report["state_agreement"]
    events = [json.loads(line) for line in (ROOT / "attempts.jsonl").read_text().splitlines()]
    for policy, suffix in [("llm", "-stop"), ("jev", "-jev")]:
        seconds = [
            e["seconds"]
            for e in events
            if e.get("status") == 200 and e.get("label", "").endswith(suffix)
        ]
        assert len(seconds) == report["latency"][policy]["n"]
        assert (
            statistics.median(seconds)
            == report["latency"][policy]["successful_request_median_seconds"]
        )
    for policy in summaries:
        lost = sum(
            r["policies"][policy]["coverage"] < r["policies"]["reference_7"]["coverage"]
            for r in expected_rows
        )
        assert lost == report["policies"][policy]["queries_losing_later_evidence"]
    cost = json.loads((SUMMARY / "decision_cost_estimate.json").read_text())
    for policy, stage in [("llm", "stop"), ("jev", "jev")]:
        selected = [
            calls[f"q{r['idx']}-r{ri}-{stage}"]
            for r in expected_rows
            for ri in range(1, r["policies"][policy]["round"] + 1)
        ]
        assert len(selected) == cost[policy]["calls"]
        for token, key in (
            [("prompt_tokens", "input_tokens"), ("completion_tokens", "output_tokens")]
            if policy == "llm"
            else [("input_tokens", "input_tokens")]
        ):
            assert sum(c["usage"][token] for c in selected) == cost[policy][key]
    assert math.isclose(
        cost["llm"]["off_peak_usd"],
        (cost["llm"]["input_tokens"] * 0.15 + cost["llm"]["output_tokens"] * 0.6) / 1e6,
    )
    assert math.isclose(cost["jev"]["usd"], cost["jev"]["input_tokens"] * 0.042 / 1e6)
    print(
        "Verified: 100 queries, 700 states, policy metrics, usage, behavior, latency and decision costs."
    )


if __name__ == "__main__":
    main()
