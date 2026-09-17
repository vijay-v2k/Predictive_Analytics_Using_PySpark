import requests
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, regexp_replace, lower, trim, expr, monotonically_increasing_id, udf
from pyspark.sql.types import ArrayType, FloatType
import psycopg2


# 1. Define a STANDARD UDF (No Pandas, No Arrow = No Java 17 crashes!)
def get_embedding(text: str):
    url = "http://host.docker.internal:1234/v1/embeddings"
    headers = {"Content-Type": "application/json"}

    # IMPORTANT: Change this to the EXACT name of the model loaded in LM Studio
    payload = {
        "input": text,
        "model": "text-embedding-nomic-embed-text-v1.5"
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        # Return the list of floats directly
        return response.json()["data"][0]["embedding"]
    except Exception as e:
        print(f"Error calling LM Studio for text '{text[:30]}...': {e}")
        return None


# Register the standard UDF
embed_udf = udf(get_embedding, ArrayType(FloatType()))


def run_real_rag_pipeline():
    print(">>> Initializing Spark Session for Real RAG Prep...")

    # No extra Java options needed anymore!
    spark = SparkSession.builder \
        .appName("BFSI_Real_RAG_Embeddings") \
        .master("local[*]") \
        .config("spark.jars.packages", "org.postgresql:postgresql:42.7.3") \
        .getOrCreate()

    jdbc_url = "jdbc:postgresql://bfsi-postgres:5432/bfsi_db"
    props = {
        "user": "bfsi_admin",
        "password": "super_secret_password",
        "driver": "org.postgresql.Driver"
    }

    # 2. Simulate ingesting raw financial text
    print(">>> Ingesting raw financial text...")
    raw_data = [
        (1, "SECTION 4.1: The Borrower shall maintain a minimum liquidity ratio of 1.25x. "
            "Failure to maintain this ratio constitutes an Event of Default, "
            "allowing the Lender to accelerate the debt and demand immediate repayment."),
        (2, "SECTION 8.3: Confidentiality. The Lender agrees to keep all financial statements, "
            "tax returns, and proprietary business models of the Borrower strictly confidential, "
            "except as required by law or regulatory authorities such as the SEC or FDIC."),
        (3, "The company experienced a significant cash flow struggle in Q3, leading to delayed vendor payments.")
    ]

    df_raw = spark.createDataFrame(raw_data, ["doc_id", "raw_text"])

    # 3. Text Cleaning & Chunking
    print(">>> Cleaning and chunking text...")
    df_cleaned = df_raw.withColumn("clean_text", lower(trim(regexp_replace(col("raw_text"), "\\s+", " "))))
    df_chunked = df_cleaned.withColumn("chunks", split(col("clean_text"), "(?<=\\.)\\s+"))
    df_exploded = df_chunked.withColumn("chunk_text", expr("explode(chunks)")).drop("chunks", "clean_text")
    df_final = df_exploded.withColumn("chunk_id", monotonically_increasing_id())

    # 4. Generate REAL Embeddings using the standard UDF
    print(">>> Calling local LM Studio for real embeddings (this may take a few seconds per chunk)...")
    df_with_vectors = df_final.withColumn("embedding", embed_udf(col("chunk_text")))

    # 5. Write to PostgreSQL with pgvector
    print(">>> Preparing PostgreSQL table...")
    conn = psycopg2.connect(host="bfsi-postgres", port="5432", database="bfsi_db", user="bfsi_admin",
                            password="super_secret_password")
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS rag_document_chunks;")

    # nomic-embed-text uses 768 dimensions
    cur.execute("""
                CREATE TABLE rag_document_chunks
                (
                    chunk_id   BIGINT PRIMARY KEY,
                    doc_id     INT,
                    chunk_text TEXT,
                    embedding  vector(768)
                );
                """)
    conn.commit()
    cur.close()
    conn.close()

    print(">>> Writing real chunks and vectors to PostgreSQL...")
    df_with_vectors.select("chunk_id", "doc_id", "chunk_text", "embedding").write.jdbc(
        url=jdbc_url,
        table="rag_document_chunks",
        mode="append",
        properties=props
    )

    print(">>> Final RAG-Ready DataFrame:")
    df_with_vectors.select("chunk_id", "doc_id", "chunk_text", "embedding").show(truncate=50)

    spark.stop()
    print(">>> Real RAG Data Prep Pipeline Complete!")


if __name__ == "__main__":
    run_real_rag_pipeline()