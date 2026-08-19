from sympy import python


python
from pathlib import Path
import pickle

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

CHUNKS_PATH = ARTIFACTS_DIR / "chunks_B.pkl"
EMBEDDINGS_PATH = ARTIFACTS_DIR / "embeddings_B.npy"


# ============================================================
# EMBEDDING MODEL
# ============================================================

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# LOAD CHUNKS
# ============================================================

def load_chunks():

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {CHUNKS_PATH}"
        )

    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    if not chunks:
        raise ValueError(
            "chunks_B.pkl is empty."
        )

    return chunks


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

def load_embeddings():

    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Embeddings file not found: {EMBEDDINGS_PATH}"
        )

    embeddings = np.load(
        EMBEDDINGS_PATH,
        allow_pickle=False
    )

    if embeddings.ndim != 2:
        raise ValueError(
            f"Invalid embeddings shape: {embeddings.shape}"
        )

    return embeddings


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

def load_embedding_model():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Embedding device: {device}")

    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME,
        device=device
    )

    return model


# ============================================================
# EXTRACT CHUNK TEXT
# ============================================================

def get_chunk_text(chunk):

    if isinstance(chunk, str):
        return chunk

    if isinstance(chunk, dict):

        for key in [
            "text",
            "content",
            "page_content"
        ]:

            if chunk.get(key):
                return str(chunk[key])

    if hasattr(chunk, "page_content"):
        return str(chunk.page_content)

    raise ValueError(
        "Cannot extract text from chunk."
    )


# ============================================================
# EXTRACT CHUNK ID
# ============================================================

def get_chunk_id(chunk, index):

    if isinstance(chunk, dict):

        for key in [
            "chunk_id",
            "id",
            "chunkId"
        ]:

            if chunk.get(key) is not None:
                return str(chunk[key])

    if hasattr(chunk, "metadata"):

        metadata = chunk.metadata

        if isinstance(metadata, dict):

            for key in [
                "chunk_id",
                "id",
                "chunkId"
            ]:

                if metadata.get(key) is not None:
                    return str(metadata[key])

    return f"chunk_{index}"


# ============================================================
# NORMALIZE EMBEDDINGS
# ============================================================

def normalize_embeddings(embeddings):

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    norms[norms == 0] = 1.0

    return embeddings / norms


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def semantic_search(
    query,
    chunks,
    embeddings,
    embedding_model,
    top_k=10
):

    if not query or not query.strip():
        raise ValueError(
            "Query cannot be empty."
        )

    if len(chunks) != len(embeddings):

        raise ValueError(
            "Chunks and embeddings count mismatch: "
            f"{len(chunks)} chunks vs "
            f"{len(embeddings)} embeddings"
        )

    top_k = min(
        int(top_k),
        len(chunks)
    )

    query_embedding = embedding_model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    normalized_embeddings = normalize_embeddings(
        embeddings
    )

    scores = (
        normalized_embeddings
        @ query_embedding
    )

    top_indices = np.argsort(
        scores
    )[::-1][:top_k]

    results = []

    for rank, index in enumerate(
        top_indices,
        start=1
    ):

        index = int(index)

        chunk = chunks[index]

        results.append(
            {
                "rank": rank,
                "chunk_id": get_chunk_id(
                    chunk,
                    index
                ),
                "text": get_chunk_text(
                    chunk
                ),
                "score": float(
                    scores[index]
                ),
                "index": index
            }
        )

    return results


# ============================================================
# CARDIOPRESS RETRIEVER
# ============================================================

class CardioPressRetriever:

    def __init__(self, top_k=10):

        self.top_k = top_k

        print(
            "Loading CardioPress retrieval artifacts..."
        )

        self.chunks = load_chunks()

        self.embeddings = load_embeddings()

        if len(self.chunks) != len(
            self.embeddings
        ):

            raise ValueError(
                "Chunks and embeddings have "
                "different lengths."
            )

        print(
            f"Chunks loaded: {len(self.chunks)}"
        )

        print(
            f"Embeddings shape: "
            f"{self.embeddings.shape}"
        )

        self.embedding_model = (
            load_embedding_model()
        )

        print(
            "CardioPress retriever ready ✓"
        )

    def search(self, query):

        return semantic_search(
            query=query,
            chunks=self.chunks,
            embeddings=self.embeddings,
            embedding_model=self.embedding_model,
            top_k=self.top_k
        )


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    retriever = CardioPressRetriever(
        top_k=10
    )

    test_query = (
        "How does reducing salt intake "
        "help prevent hypertension?"
    )

    results = retriever.search(
        test_query
    )

    print()
    print("=" * 80)
    print("RETRIEVAL TEST")
    print("=" * 80)

    print(
        f"Retrieved chunks: {len(results)}"
    )

    for result in results:

        print(
            f"Rank {result['rank']} | "
            f"{result['chunk_id']} | "
            f"Score: {result['score']:.4f}"
        )

        print(
            result["text"][:250]
        )

        print("-" * 80)

