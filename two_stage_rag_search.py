import requests
import psycopg2
from sentence_transformers import CrossEncoder


def two_stage_search(query: str, top_k_vector: int = 5, top_k_rerank: int = 2):
    print(f"\n🔍 Stage 1: Vector Search for '{query}'")

    # 1. Embed the query (Bi-Encoder via LM Studio)
    embed_url = "http://host.docker.internal:1234/v1/embeddings"
    embed_payload = {
        "input": query,
        "model": "text-embedding-nomic-embed-text-v1.5"  # <-- Ensure this matches your loaded model
    }

    try:
        response = requests.post(embed_url, json=embed_payload, timeout=10)
        response.raise_for_status()
        query_vector = response.json()["data"][0]["embedding"]
    except Exception as e:
        print(f"❌ Error calling LM Studio for embedding: {e}")
        return

    vector_str = "[" + ",".join(map(str, query_vector)) + "]"

    # 2. Fetch top K candidates from pgvector (Fast, but "dumb")
    conn = psycopg2.connect(host="bfsi-postgres", port="5432", database="bfsi_db", user="bfsi_admin",
                            password="super_secret_password")
    cur = conn.cursor()

    sql = f"""
        SELECT chunk_id, doc_id, chunk_text, 
               (embedding <=> '{vector_str}'::vector) AS distance
        FROM rag_document_chunks
        ORDER BY distance ASC
        LIMIT {top_k_vector};
    """
    cur.execute(sql)
    vector_results = cur.fetchall()

    print(f"   ✅ Retrieved top {len(vector_results)} candidates from pgvector.")
    documents_to_rerank = [row[2] for row in vector_results]

    # 3. Stage 2: Cross-Encoder Reranking (Running locally in Python!)
    print(f"\n🧠 Stage 2: Cross-Encoder Reranking top {len(documents_to_rerank)} documents...")
    print("   (Loading local cross-encoder model... this may take a moment on first run)")

    # Industry standard lightweight reranker (only ~90MB, downloads automatically on first run)
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # Format pairs for the cross-encoder: [(query, doc1), (query, doc2), ...]
    pairs = [[query, doc] for doc in documents_to_rerank]

    # Get relevance scores (higher is better, typically between -10 and 10)
    scores = reranker.predict(pairs)

    # Combine scores with original database rows
    scored_results = list(zip(scores, vector_results))

    # Sort by score descending (highest relevance first)
    scored_results.sort(key=lambda x: x[0], reverse=True)

    print(f"\n🏆 FINAL RERANKED RESULTS (Top {top_k_rerank}):")
    print("-" * 90)

    for i, (score, row) in enumerate(scored_results[:top_k_rerank], 1):
        chunk_id, doc_id, chunk_text, distance = row
        print(f"Rank {i} | Relevance Score: {score:>.4f} | Vector Distance: {distance:>.4f} | Doc ID: {doc_id}")
        print(f"Text: '{chunk_text}'\n")
        print("-" * 90)

    cur.close()
    conn.close()


if __name__ == "__main__":
    # A tricky financial query where nuance matters
    two_stage_search(
        query="Are we allowed to share our financial models with the SEC?",
        top_k_vector=5,
        top_k_rerank=2
    )