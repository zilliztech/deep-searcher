"""Rebuild the frozen corpus index without making paid API requests."""

import gzip
import json
import subprocess
import sys

import experiment_common as p
import numpy as np
from pymilvus import MilvusClient


def main():
    p.ROOT.mkdir(parents=True, exist_ok=True)
    with gzip.open(p.HERE / "artifacts/fixtures/corpus.json.gz", "rt") as handle:
        corpus = json.load(handle)
    (p.ROOT / "corpus.json").write_text(json.dumps(corpus))
    subprocess.run([sys.executable, str(p.HERE / "bge_worker.py")], input="", text=True, check=True)
    vectors = np.load(p.ROOT / "bge_corpus.npy")
    assert len(corpus) == len(vectors) == 6119
    client = MilvusClient(str(p.ROOT / "milvus.db"))
    try:
        if client.has_collection("bge_pilot"):
            assert client.get_collection_stats("bge_pilot")["row_count"] == len(corpus)
            return
        client.create_collection("bge_pilot", dimension=1024, metric_type="COSINE")
        for start in range(0, len(corpus), 200):
            client.insert(
                "bge_pilot",
                [
                    {"id": i, "vector": vectors[i].tolist(), **corpus[i]}
                    for i in range(start, min(start + 200, len(corpus)))
                ],
            )
    finally:
        client.close()


if __name__ == "__main__":
    main()
