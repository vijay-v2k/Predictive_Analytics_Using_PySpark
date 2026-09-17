FROM python:3.11-slim

# Install Java (Required for PySpark) and system dependencies
RUN apt-get update && apt-get install -y default-jdk curl && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PYSPARK_PYTHON=python3

# Install the Universal Data Science, Engineering, and API Stack
RUN pip install --no-cache-dir \
    pyspark==3.5.1 \
    mlflow==2.16.2 \
    psycopg2-binary \
    fastapi uvicorn pydantic \
    requests sentence-transformers \
    grpcio grpcio-status \
    scikit-learn pandas numpy xgboost lightgbm \
    matplotlib seaborn plotly \
    jupyter

WORKDIR /workspace

CMD ["tail", "-f", "/dev/null"]