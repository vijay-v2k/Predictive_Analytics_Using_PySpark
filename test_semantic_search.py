import requests
import psycopg2


def search_rag(query: str):
    print(f"\n🔍 Searching for: '{query}'")

    # 1. Embed the user's query using LM Studio on the Windows Host
    # host.docker.internal is the magic DNS name that points from Docker to Windows
    url = "http://host.docker.internal:1234/v1/embeddings"
    payload = {
        "input": query,
        "model": "nomic-embed-text"  # <-- MUST MATCH YOUR LM STUDIO MODEL NAME EXACTLY
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        query_vector = response.json()["data"][0]["embedding"]
    except Exception as e:
        print(f"❌ Error calling LM Studio: {e}")
        print("💡 Tip: Make sure LM Studio Local Server is running on port 1234!")
        return

    # 2. Format vector as a string for PostgreSQL
    vector_str = "[" + ",".join(map(str, query_vector)) + "]"

    # 3. Query Postgres using the internal Docker network name and port
    conn = psycopg2.connect(
        host="bfsi-postgres",  # Internal Docker network name
        port="5432",  # Internal Docker port
        database="bfsi_db",
        user="bfsi_admin",
        password="super_secret_password"
    )
    cur = conn.cursor()

    # <=> is the pgvector cosine distance operator. Lower = more similar!
    sql = f"""
        SELECT chunk_id, doc_id, chunk_text, 
               (embedding <=> '{vector_str}'::vector) AS distance
        FROM rag_document_chunks
        ORDER BY distance ASC
        LIMIT 2;
    """
    cur.execute(sql)
    results = cur.fetchall()

    print("✅ Top Matches:")
    for i, row in enumerate(results, 1):
        print(f"  {i}. Distance: {row[3]:.4f} | Doc {row[1]}")
        print(f"     Text: '{row[2]}'\n")

    cur.close()
    conn.close()


if __name__ == "__main__":
    # Test 1: Semantic match for "cash flow struggle" (uses different words)
    search_rag("What happens if the company has no money to pay vendors?")

    # Test 2: Semantic match for "confidentiality" and "tax returns"
    search_rag("Can the bank see our private tax documents?")