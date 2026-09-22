"""Audit every complete trajectory before publishing the experiment summary."""

import hashlib
import json
import statistics
from collections import Counter
from datetime import datetime, timezone

import experiment_common as p
from full100_core import atomic_json
from run_full100 import MAX_ROUNDS, OUT, clean_text, parse_indices


def finalize():
    manifest = json.loads((OUT / "manifest.json").read_text())
    calls = {f.stem: json.loads(f.read_text()) for f in (OUT / "calls").glob("*.json")}
    corpus = {d["text"]: d for d in json.loads((p.ROOT / "corpus.json").read_text())}
    assert manifest["source_sha256"] == hashlib.sha256(p.source.encode()).hexdigest()
    rows = []
    all_states = []
    for qi, qid in zip(manifest["indices"], manifest["ids"]):
        trace = json.loads((OUT / "traces" / f"{qi}.json").read_text())
        assert trace["id"] == qid and len(trace["rounds"]) == MAX_ROUNDS
        query, gold = trace["query"], set(trace["gold_titles"])
        contexts, cumulative = [], []

        def record(stage, ri):
            call = calls[f"q{qi}-r{ri}-{stage}"]
            return json.loads((OUT / "cache" / (call["digest"] + ".json")).read_text())

        for r in trace["rounds"]:
            ri = r["round"]
            sub = record("subquery", ri)
            assert sub["request"]["messages"][0]["content"] == p.PROMPTS[
                "FOLLOWUP_QUERY_PROMPT"
            ].format(query=query, intermediate_context="\n".join(contexts))
            assert clean_text(sub["response"]) == r["subquery"]
            for d in r["retrieved"]:
                assert d["metadata"] == corpus[d["text"]]["metadata"]
            ans = record("answer", ri)
            assert ans["request"]["messages"][0]["content"] == p.PROMPTS[
                "INTERMEDIATE_ANSWER_PROMPT"
            ].format(retrieved_documents=p.format_docs(r["retrieved"]), sub_query=r["subquery"])
            assert clean_text(ans["response"]) == r["answer"]
            if "No relevant information found" not in r["answer"]:
                sel = record("selection", ri)
                assert sel["request"]["messages"][0]["content"] == p.PROMPTS[
                    "GET_SUPPORTED_DOCS_PROMPT"
                ].format(
                    retrieved_documents=p.format_docs(r["retrieved"]),
                    query=r["subquery"],
                    answer=r["answer"],
                )
                indices = parse_indices(clean_text(sel["response"]), len(r["retrieved"]))
                assert r["selected"] == [r["retrieved"][i] for i in indices]
            else:
                assert not r["selected"]
            cumulative = p.unique(cumulative + r["selected"])
            assert p.metrics(cumulative, gold) == r["metrics"]
            contexts.append(
                f"Intermediate query{ri}: {r['subquery']}\nIntermediate answer{ri}: {r['answer']}"
            )
            stop, jev = record("stop", ri), record("jev", ri)
            assert stop["request"]["messages"][0]["content"] == p.PROMPTS[
                "REFLECTION_PROMPT"
            ].format(query=query, intermediate_context="\n".join(contexts))
            assert clean_text(stop["response"]).lower() == ("yes" if r["llm_stop"] else "no")
            assert jev["request"]["state"] == {
                "main_query": query,
                "intermediate_context": "\n".join(contexts),
            }
            assert jev["request"]["questions"]["enough_information"] == p.QUESTION
            assert jev["response"]["answers"]["enough_information"]["noul"] == r["jev_score"]
            all_states.append(r)
        policies = {}
        for policy in ["llm", "jev"]:
            candidates = [
                r
                for r in trace["rounds"]
                if (r["llm_stop"] if policy == "llm" else r["jev_score"] >= 0.5)
            ]
            chosen = candidates[0] if candidates else trace["rounds"][-1]
            policies[policy] = {
                "round": chosen["round"],
                "hit_limit_without_stop": not candidates,
                **chosen["metrics"],
            }
        policies["reference_7"] = {"round": 7, **trace["rounds"][-1]["metrics"]}
        rows.append(
            {
                "idx": qi,
                "id": qid,
                "query": query,
                "type": trace["type"],
                "gold_titles": sorted(gold),
                "policies": policies,
            }
        )
    assert len(rows) == 100
    assert dict(Counter(r["type"] for r in rows)) == manifest["type_counts"]
    summary = {}
    for policy in ["llm", "jev", "reference_7"]:
        summary[policy] = {
            metric: statistics.mean(r["policies"][policy][metric] for r in rows)
            for metric in ["round", "r2", "r5", "coverage", "all_supports"]
        }
        summary[policy]["queries_losing_later_evidence"] = sum(
            r["policies"][policy]["coverage"] < r["policies"]["reference_7"]["coverage"]
            for r in rows
        )
        selected_calls = []
        for row in rows:
            for ri in range(1, row["policies"][policy]["round"] + 1):
                for stage in ["subquery", "answer", "selection"] + (
                    ["stop"] if policy == "llm" else ["jev"] if policy == "jev" else []
                ):
                    label = f"q{row['idx']}-r{ri}-{stage}"
                    if label in calls:
                        selected_calls.append(calls[label])
        summary[policy]["llm_requests"] = sum(c["kind"] == "llm" for c in selected_calls)
        summary[policy]["jev_requests"] = sum(c["kind"] == "jev" for c in selected_calls)
        summary[policy]["llm_input_tokens"] = sum(
            c["usage"].get("prompt_tokens", 0) for c in selected_calls if c["kind"] == "llm"
        )
        summary[policy]["llm_output_tokens"] = sum(
            c["usage"].get("completion_tokens", 0) for c in selected_calls if c["kind"] == "llm"
        )
        summary[policy]["jev_input_tokens"] = sum(
            c["usage"].get("input_tokens", 0) for c in selected_calls if c["kind"] == "jev"
        )
        summary[policy]["jev_estimated_usd"] = summary[policy]["jev_input_tokens"] * 0.042 / 1e6
    behavior = Counter(
        "same"
        if r["policies"]["jev"]["round"] == r["policies"]["llm"]["round"]
        else "jev_earlier"
        if r["policies"]["jev"]["round"] < r["policies"]["llm"]["round"]
        else "jev_later"
        for r in rows
    )
    events = [
        json.loads(line)
        for line in (OUT / "attempts.jsonl").read_text().splitlines()
        if line.strip()
    ]
    usage = {
        "llm_attempts": sum(e["kind"] == "llm" for e in events),
        "jev_attempts": sum(e["kind"] == "jev" for e in events),
        "http_429": sum(e.get("status") == 429 for e in events),
        "transport_errors": sum(e.get("status") == "transport_error" for e in events),
        "invalid_outputs": sum(e.get("status") == "invalid_output" for e in events),
        "jev_input_tokens_new": sum(
            e.get("usage", {}).get("input_tokens", 0) for e in events if e["kind"] == "jev"
        ),
    }
    usage["jev_estimated_usd_new"] = usage["jev_input_tokens_new"] * 0.042 / 1e6
    latencies = {}
    for policy, suffix in [("llm", "-stop"), ("jev", "-jev")]:
        values = [
            e["seconds"]
            for e in events
            if e.get("status") == 200 and e.get("label", "").endswith(suffix)
        ]
        latencies[policy] = {
            "n": len(values),
            "successful_request_median_seconds": statistics.median(values) if values else None,
        }
    report = {
        "n": 100,
        "policies": summary,
        "first_stop_behavior": dict(behavior),
        "state_agreement": statistics.mean(
            r["llm_stop"] == (r["jev_score"] >= 0.5) for r in all_states
        ),
        "new_api_usage": usage,
        "latency": latencies,
        "validation": {
            "queries": 100,
            "states": 700,
            "input_alignment": True,
            "document_integrity_and_order": True,
            "metrics_recomputed": True,
        },
        "limitations": [
            "Evidence retrieval metrics, not final-answer correctness.",
            "Paired shared-trajectory replay, not separate end-to-end latency runs.",
            "BGE index differs from historical ada-002 evaluation.",
            "Company API billing is not estimated. Jev price reference is $0.042 per million input tokens.",
        ],
    }
    atomic_json(OUT / "results.json", rows)
    atomic_json(OUT / "report.json", report)
    lines = [
        "# DeepSearcher stopping evaluation: 100 queries",
        "",
        "| Policy | Mean rounds | Recall@2 | Recall@5 | Evidence coverage | Queries losing later evidence |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, value in summary.items():
        lines.append(
            f"| {name} | {value['round']:.2f} | {value['r2']:.3f} | {value['r5']:.3f} | {value['coverage']:.3f} | {value['queries_losing_later_evidence']} |"
        )
    lines.extend(
        [
            "",
            "## First stopping position",
            "",
            json.dumps(dict(behavior)),
            "",
            "## Usage and validation",
            "",
            f"New Jev estimated cost: ${usage['jev_estimated_usd_new']:.6f}. All 700 states passed input, document, ordering, and metric audits.",
            "",
            "## Limitations",
            "",
        ]
        + ["- " + x for x in report["limitations"]]
    )
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    status = json.loads((OUT / "status.json").read_text())
    status.update(
        state="complete",
        updated_at=datetime.now(timezone.utc).isoformat(),
        audited_queries=100,
        audited_rounds=700,
    )
    atomic_json(OUT / "status.json", status)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        finalize()
    except Exception as exc:
        atomic_json(
            OUT / "audit_failure.json",
            {
                "error": type(exc).__name__,
                "message": "Audit failed; do not publish a completed result.",
            },
        )
        status = json.loads((OUT / "status.json").read_text())
        status["state"] = "needs_attention"
        atomic_json(OUT / "status.json", status)
        raise
