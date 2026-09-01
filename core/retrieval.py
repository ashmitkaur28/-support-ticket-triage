
import argparse
import pickle
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

KB_DIR = Path(__file__).parent.parent / "data" / "knowledge_base"
DB_DIR = Path(__file__).parent.parent / "chroma_db"
BM25_PATH = Path(__file__).parent.parent / "chroma_db" / "bm25_index.pkl"

DEFAULT_EF = embedding_functions.DefaultEmbeddingFunction()


def get_collection():
    client = chromadb.PersistentClient(path=str(DB_DIR))
    return client.get_or_create_collection(
        name="support_kb",
        embedding_function=DEFAULT_EF,
    )


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def build_index():
    collection = get_collection()
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    docs, ids, metadatas = [], [], []
    for path in sorted(KB_DIR.glob("*.md")):
        text = path.read_text()
        chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
        for i, chunk in enumerate(chunks):
            docs.append(chunk)
            ids.append(f"{path.stem}-{i}")
            metadatas.append({"source": path.name})

    # semantic index (embeddings)
    collection.add(documents=docs, ids=ids, metadatas=metadatas)

    # keyword index (BM25) — stored alongside, keyed by the same ids
    tokenized = [_tokenize(d) for d in docs]
    bm25 = BM25Okapi(tokenized)
    DB_DIR.mkdir(exist_ok=True)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "docs": docs, "ids": ids, "metadatas": metadatas}, f)

    print(f"Indexed {len(docs)} chunks from {len(list(KB_DIR.glob('*.md')))} docs (semantic + keyword)")


def _semantic_search(query: str, k: int) -> dict:
    collection = get_collection()
    results = collection.query(query_texts=[query], n_results=k)
    return {
        cid: {"text": doc, "source": meta["source"], "rank": rank}
        for rank, (cid, doc, meta) in enumerate(zip(
            results["ids"][0], results["documents"][0], results["metadatas"][0]
        ))
    }


def _keyword_search(query: str, k: int) -> dict:
    with open(BM25_PATH, "rb") as f:
        data = pickle.load(f)
    scores = data["bm25"].get_scores(_tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return {
        data["ids"][i]: {"text": data["docs"][i], "source": data["metadatas"][i]["source"], "rank": rank}
        for rank, i in enumerate(ranked)
    }


def retrieve(query: str, k: int = 3) -> list[dict]:
   
    semantic = _semantic_search(query, k=k * 2)
    keyword = _keyword_search(query, k=k * 2)

    # reciprocal rank fusion: combine scores from both rankings
    RRF_K = 60  # standard constant, dampens the effect of rank 1 vs rank 2
    fused_scores: dict[str, float] = {}
    all_chunks: dict[str, dict] = {}

    for cid, info in semantic.items():
        fused_scores[cid] = fused_scores.get(cid, 0) + 1 / (RRF_K + info["rank"])
        all_chunks[cid] = info

    for cid, info in keyword.items():
        fused_scores[cid] = fused_scores.get(cid, 0) + 1 / (RRF_K + info["rank"])
        all_chunks.setdefault(cid, info)

    top_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)[:k]
    return [
        {"text": all_chunks[cid]["text"], "source": all_chunks[cid]["source"], "fused_score": fused_scores[cid]}
        for cid in top_ids
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--query", type=str)
    args = parser.parse_args()

    if args.build:
        build_index()
    elif args.query:
        for hit in retrieve(args.query):
            print(f"[{hit['source']}] (score={hit['fused_score']:.4f})\n{hit['text']}\n")
    else:
        parser.print_help()