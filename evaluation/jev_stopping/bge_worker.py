"""Local BGE embedding worker; JSON lines on stdin/stdout."""

import json
import os
import sys

import experiment_common as p
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

root = p.ROOT
torch.set_num_threads(4)
model_path = os.environ.get("BGE_MODEL_PATH", "BAAI/bge-large-en-v1.5")
revision = "d4aa6901d3a41ba39fb536a557fa166f842b0e09"
device = os.environ.get("BGE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)
model = AutoModel.from_pretrained(model_path, revision=revision).to(device).eval()


def encode(texts):
    inputs = tokenizer(
        texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
    ).to(device)
    with torch.inference_mode():
        vectors = model(**inputs).last_hidden_state[:, 0]
        return torch.nn.functional.normalize(vectors, p=2, dim=1).float().cpu().numpy()


path = root / "bge_corpus.npy"
if not path.exists():
    docs = json.loads((root / "corpus.json").read_text())
    vectors = []
    for i in range(0, len(docs), 24):
        vectors.append(encode([d["text"] for d in docs[i : i + 24]]))
        if i % 480 == 0:
            print(f"Embedded {i}/{len(docs)} documents", file=sys.stderr, flush=True)
    np.save(path, np.concatenate(vectors))
    lengths = [
        len(tokenizer(d["text"], truncation=False, verbose=False)["input_ids"]) for d in docs
    ]
    (root / "embedding_manifest.json").write_text(
        json.dumps(
            {
                "model": "BAAI/bge-large-en-v1.5",
                "snapshot": revision,
                "pooling": "CLS",
                "normalized": True,
                "max_length": 512,
                "documents": len(docs),
                "truncated_documents": sum(n > 512 for n in lengths),
                "query_prefix": "Represent this sentence for searching relevant passages: ",
            },
            indent=2,
        )
    )
print(json.dumps({"ready": True}), flush=True)
for line in sys.stdin:
    query = json.loads(line)["query"]
    vector = encode(["Represent this sentence for searching relevant passages: " + query])[0]
    print(json.dumps({"vector": vector.tolist()}), flush=True)
