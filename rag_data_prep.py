from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, regexp_replace, lower, trim, expr, monotonically_increasing_id, udf
from pyspark.sql.types import ArrayType, FloatType
import random
import psycopg2

# UDF to generate a native list of floats (not a string)
def generate_mock_vector(dimensions=10):
    return [round(random.uniform(-0.1, 0.1), 4) for _ in range(dimensions)]

vector_udf = udf(lambda: generate_mock_vector(10), ArrayType(FloatType()))

def run_rag_pipeline():
    print(">>> Initializing Spark Session for RAG Prep...")

    spark = SparkSession.builder \
        .appName("BFSI_RAG_Data_Prep") \
        .master("local[*]") \
        .config("spark.jars.packages", "org.postgresql:postgresql:42.7.3") \
        .getOrCreate()

    jdbc_url = "jdbc:postgresql://bfsi-postgres:5432/bfsi_db"
    props = {
        "user": "bfsi_admin",
        "password": "super_secret_password",
        "driver": "org.postgresql.Driver"
    }

    # 1. Simulate ingesting raw, messy financial text
    print(">>> Ingesting raw financial text...")
    raw_data = [
        (1, "SECTION 4.1: The Borrower shall maintain a minimum liquidity ratio of 1.25x. "
            "Failure to maintain this ratio constitutes an Event of Default, "
            "allowing the Lender to accelerate the debt and demand immediate repayment of all outstanding principal and accrued interest."),
        (2, "SECTION 8.3: Confidentiality. The Lender agrees to keep all financial statements, "
            "tax returns, and proprietary business models of the Borrower strictly confidential, "
            "except as required by law or regulatory authorities such as the SEC or FDIC.")
    ]

    df_raw = spark.createDataFrame(raw_data, ["doc_id", "raw_text"])

    # 2. Text Cleaning & Chunking
    print(">>> Cleaning and chunking text...")
    df_cleaned = df_raw.withColumn("clean_text",
        lower(trim(regexp_replace(col("raw_text"), "\\s+", " "))) # Normalize whitespace and lowercase
    )

    # Split text into chunks by period + space
    df_chunked = df_cleaned.withColumn("chunks", split(col("clean_text"), "(?<=\\.)\\s+"))

    # Explode the array of chunks into separate rows (1 row per chunk)
    df_exploded = df_chunked.withColumn("chunk_text", expr("explode(chunks)")).drop("chunks", "clean_text")

    # Add a unique chunk ID
    df_final = df_exploded.withColumn("chunk_id", monotonically_increasing_id())

    # 3. Generate Embeddings as a NATIVE Float Array
    print(">>> Generating vector embeddings (as Float Array)...")
    df_with_vectors = df_final.withColumn("embedding", vector_udf())

    # 4. Write to PostgreSQL with pgvector
    print(">>> Preparing PostgreSQL table...")
    conn = psycopg2.connect(host="bfsi-postgres", port="5432", database="bfsi_db", user="bfsi_admin", password="super_secret_password")
    cur = conn.cursor()

    # Drop table if exists to ensure a clean schema with the correct vector(10) type
    cur.execute("DROP TABLE IF EXISTS rag_document_chunks;")

    cur.execute("""
        CREATE TABLE rag_document_chunks (
            chunk_id BIGINT PRIMARY KEY,
            doc_id INT,
            chunk_text TEXT,
            embedding vector(10) 
        );
    """)
    conn.commit()
    cur.close()
    conn.close()

    print(">>> Writing chunks and vectors to PostgreSQL...")
    # Because 'embedding' is now ArrayType(FloatType), JDBC sends it as float4[],
    # which pgvector automatically accepts and casts to 'vector'.
    df_with_vectors.select("chunk_id", "doc_id", "chunk_text", "embedding").write.jdbc(
        url=jdbc_url,
        table="rag_document_chunks",
        mode="append",
        properties=props
    )

    print(">>> Final RAG-Ready DataFrame:")
    df_with_vectors.select("chunk_id", "doc_id", "chunk_text", "embedding").show(truncate=50)

    spark.stop()
    print(">>> RAG Data Prep Pipeline Complete!")

if __name__ == "__main__":
    run_rag_pipeline()