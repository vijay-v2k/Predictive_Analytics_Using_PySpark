from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pyspark.sql import SparkSession
from sentence_transformers import CrossEncoder
import requests
import time
import mlflow
import json

app = FastAPI(title="BFSI RAG API", version="1.0.0")

# Configure MLflow
mlflow.set_tracking_uri("http://bfsi-mlflow:5000")
mlflow.set_experiment("BFSI_RAG_API_Audit_Logs")

# Global variables for lazy initialization
spark = None
reranker = None


def get_spark():
    global spark
    if spark is None:
        print(">>> Initializing Spark Connect Session...")
        max_retries = 5
        for attempt in range(max_retries):
            try:
                spark = SparkSession.builder \
                    .remote("sc://spark-connect:15002") \
                    .appName("FastAPI_BFSI_Client") \
                    .config("spark.connect.timeout", "120s") \
                    .config("spark.sql.connect.timeout", "120s") \
                    .getOrCreate()
                spark.sql("SELECT 1").collect()
                print(">>> Spark Connect successfully initialized!")
                break
            except Exception as e:
                print(f">>> Spark Connect attempt {attempt + 1} failed. Retrying in 3s... ({e})")
                time.sleep(3)
        else:
            raise RuntimeError("Failed to connect to Spark Connect server.")
    return spark


def get_reranker():
    global reranker
    if reranker is None:
        print(">>> Loading Cross-Encoder Reranker model (one-time cost)...")
        # Industry standard lightweight reranker
        reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        print(">>> Reranker loaded successfully!")
    return reranker


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "BFSI RAG API"}


@app.post("/api/v1/rag/search")
def rag_search(request: SearchRequest):
    with mlflow.start_run(run_name=f"rag_search_{int(time.time())}") as run:
        try:
            # 1. Log input parameters for SR 11-7 Compliance
            mlflow.log_param("user_query", request.query)
            mlflow.log_param("top_k_requested", request.top_k)
            mlflow.log_param("embedding_model", "nomic-ai/nomic-embed-text-v1.5-GGUF")
            mlflow.log_param("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")

            # 2. Stage 1: Get Embedding from LM Studio
            embed_url = "http://host.docker.internal:1234/v1/embeddings"
            response = requests.post(embed_url, json={
                "input": request.query,
                "model": "nomic-ai/nomic-embed-text-v1.5-GGUF"
            }, timeout=10)
            response.raise_for_status()
            query_vector = response.json()["data"][0]["embedding"]
            vector_str = "[" + ",".join(map(str, query_vector)) + "]"

            # 3. Stage 1: Query PostgreSQL via Spark Connect
            spark_session = get_spark()
            jdbc_url = "jdbc:postgresql://bfsi-postgres:5432/bfsi_db"
            props = {
                "user": "bfsi_admin",
                "password": "super_secret_password",
                "driver": "org.postgresql.Driver"
            }

            # Fetch slightly more than top_k to give the reranker a wider net (e.g., top_k * 2)
            fetch_limit = request.top_k * 2
            subquery = f"""
                (SELECT chunk_id, doc_id, chunk_text, 
                        (embedding <=> '{vector_str}'::vector) AS distance 
                 FROM rag_document_chunks 
                 ORDER BY distance ASC 
                 LIMIT {fetch_limit}) AS subq
            """

            df = spark_session.read.jdbc(url=jdbc_url, table=subquery, properties=props)
            raw_results = df.collect()

            # 4. Stage 2: Cross-Encoder Reranking
            reranker_model = get_reranker()

            # Format pairs for the cross-encoder: [[query, doc1], [query, doc2], ...]
            pairs = [[request.query, row.chunk_text] for row in raw_results]

            # Get relevance scores
            scores = reranker_model.predict(pairs)

            # Combine scores with original data
            scored_results = list(zip(scores, raw_results))

            # Sort by relevance score descending (highest first)
            scored_results.sort(key=lambda x: x[0], reverse=True)

            # Take only the final top_k requested by the user
            final_results = scored_results[:request.top_k]

            # 5. Format the final output with Rank and Relevance Score
            formatted_results = []
            for rank, (score, row) in enumerate(final_results, start=1):
                formatted_results.append({
                    "rank": rank,
                    "relevance_score": float(score),
                    "chunk_id": int(row.chunk_id),
                    "doc_id": int(row.doc_id),
                    "chunk_text": row.chunk_text,
                    "vector_distance": float(row.distance)
                })

            # 6. Log metrics and artifacts for Compliance
            mlflow.log_metric("results_returned", len(formatted_results))
            if formatted_results:
                mlflow.log_metric("best_relevance_score", formatted_results[0]["relevance_score"])
                mlflow.log_metric("best_vector_distance", formatted_results[0]["vector_distance"])

            mlflow.log_text(json.dumps(formatted_results, indent=2), "reranked_context.json")

            return {
                "query": request.query,
                "results": formatted_results,
                "count": len(formatted_results),
                "mlflow_run_id": run.info.run_id
            }

        except Exception as e:
            mlflow.log_param("error", str(e))
            raise HTTPException(status_code=500, detail=str(e))