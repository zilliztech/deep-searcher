"""Frozen experiment prompts, configuration and evidence metrics."""

import ast
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("EXPERIMENT_DIR", str(HERE / "runs" / "full100"))).resolve()
SOURCE = HERE.parents[1]
CACHE = ROOT / "legacy-cache"
MODEL = os.environ.get("LLM_MODEL", "deepseek-v4-flash")
BASE = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
source = (HERE / "fixtures" / "chain_of_rag.py").read_text()
PROMPTS = {
    n.targets[0].id: ast.literal_eval(n.value)
    for n in ast.parse(source).body
    if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
}
QUESTION = {
    "type": "noul",
    "instructions": "Do the intermediate questions and answers contain enough information to answer the main query completely, without another search?",
    "criteria": {
        "true": "All facts needed to answer the main query are established. For A -> B -> C, both links must be known. A comparison requires the relevant fact for both entities.",
        "false": "At least one required fact or link is missing, unresolved, or contradictory. Knowing only A -> B is insufficient when the query asks about C. A partial answer is insufficient.",
    },
}


def unique(docs):
    seen = set()
    output = []
    for doc in docs:
        if doc["text"] not in seen:
            seen.add(doc["text"])
            output.append(doc)
    return output


def format_docs(docs):
    return "\n".join(
        f"<Document {i}>\n{d['metadata'].get('wider_text', d['text'])}\n<\\Document {i}>"
        for i, d in enumerate(docs)
    )


def metrics(docs, gold):
    titles = [d["metadata"]["title"] for d in docs]
    return {
        "r2": len(set(titles[:2]) & gold) / len(gold),
        "r5": len(set(titles[:5]) & gold) / len(gold),
        "coverage": len(set(titles) & gold) / len(gold),
        "all_supports": int(gold <= set(titles)),
        "titles": titles,
    }
