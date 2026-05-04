"""
RAG Embedder
=============
Converts patient records into embeddings and stores them in a
ChromaDB vector database for semantic retrieval.
"""

import os, sys
import pandas as pd
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJ_ROOT / "data" / "heart.csv"
CHROMA_DIR = PROJ_ROOT / "rag" / "chroma_db"


def build_vector_store():
    """Embed patient notes and store in ChromaDB."""
    print("=" * 60)
    print("  RAG EMBEDDER – Building Vector Store")
    print("=" * 60)

    # Import here to allow graceful failure messages
    from sentence_transformers import SentenceTransformer
    import chromadb

    # Load data
    df = pd.read_csv(DATA_PATH)
    print(f"\nLoaded {len(df)} patient records.")

    if "patient_note" not in df.columns:
        print("ERROR: 'patient_note' column not found. Run data/prepare_data.py first.")
        sys.exit(1)

    # Initialize embedding model
    print("\nLoading embedding model (all-MiniLM-L6-v2) …")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Generate embeddings
    print("Generating embeddings …")
    notes = df["patient_note"].tolist()
    embeddings = embed_model.encode(notes, show_progress_bar=True, batch_size=32)
    print(f"  → Generated {len(embeddings)} embeddings (dim={embeddings.shape[1]})")

    # Store in ChromaDB
    print("\nStoring in ChromaDB …")
    os.makedirs(CHROMA_DIR, exist_ok=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Delete existing collection if exists
    try:
        client.delete_collection("patients")
    except Exception:
        pass

    collection = client.create_collection(
        name="patients",
        metadata={"hnsw:space": "cosine"}
    )

    # Add documents in batches
    batch_size = 50
    for i in range(0, len(df), batch_size):
        end = min(i + batch_size, len(df))
        batch_ids = df["patient_id"].iloc[i:end].tolist()
        batch_docs = notes[i:end]
        batch_embeds = embeddings[i:end].tolist()
        batch_meta = []
        for _, row in df.iloc[i:end].iterrows():
            batch_meta.append({
                "patient_id": str(row["patient_id"]),
                "age": int(row["age"]),
                "sex": int(row["sex"]),
                "target": int(row["target"]),
                "bmi": float(row["bmi"]),
                "stress_level": float(row["stress_level"]),
                "anxiety_score": int(row["anxiety_score"]),
            })

        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            embeddings=batch_embeds,
            metadatas=batch_meta,
        )

    print(f"✓ Stored {collection.count()} patient records in ChromaDB.")
    print(f"  Database path: {CHROMA_DIR}")
    return collection


if __name__ == "__main__":
    build_vector_store()
