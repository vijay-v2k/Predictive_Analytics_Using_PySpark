import psycopg2
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sha2, concat


def seed_database():
    """Uses psycopg2 to create a raw table with sensitive PII."""
    print(">>> Seeding PostgreSQL with raw PII data...")
    # Note: We use 'bfsi-postgres' because this script runs INSIDE the Docker network
    conn = psycopg2.connect(
        host="bfsi-postgres",
        database="bfsi_db",
        user="bfsi_admin",
        password="super_secret_password"
    )
    cur = conn.cursor()

    # Create table
    cur.execute("""
                CREATE TABLE IF NOT EXISTS raw_customer_data
                (
                    id
                    SERIAL
                    PRIMARY
                    KEY,
                    first_name
                    VARCHAR
                (
                    50
                ),
                    last_name VARCHAR
                (
                    50
                ),
                    ssn VARCHAR
                (
                    11
                ),
                    salary INT
                    );
                """)

    # Insert dummy data if the table is empty
    cur.execute("SELECT COUNT(*) FROM raw_customer_data;")
    if cur.fetchone()[0] == 0:
        cur.execute("""
                    INSERT INTO raw_customer_data (first_name, last_name, ssn, salary)
                    VALUES ('John', 'Doe', '123-45-6789', 85000),
                           ('Jane', 'Smith', '987-65-4321', 92000),
                           ('Alice', 'Johnson', '456-78-9012', 78000);
                    """)
        conn.commit()
        print(">>> Raw data inserted.")
    else:
        print(">>> Raw data already exists.")

    cur.close()
    conn.close()


def run_spark_etl():
    """Uses PySpark to read, mask PII, and write back to Postgres."""
    print(">>> Initializing Spark Session (Downloading JDBC driver on first run)...")

    # Initialize Spark and tell it to download the Postgres JDBC driver from Maven
    spark = SparkSession.builder \
        .appName("BFSI_PII_Masking_Pipeline") \
        .master("local[*]") \
        .config("spark.jars.packages", "org.postgresql:postgresql:42.7.3") \
        .getOrCreate()

    jdbc_url = "jdbc:postgresql://bfsi-postgres:5432/bfsi_db"
    connection_properties = {
        "user": "bfsi_admin",
        "password": "super_secret_password",
        "driver": "org.postgresql.Driver"
    }

    # 1. Read raw PII data from Postgres
    print(">>> Reading raw data into Spark DataFrame...")
    df_raw = spark.read.jdbc(
        url=jdbc_url,
        table="raw_customer_data",
        properties=connection_properties
    )

    # 2. Mask PII (Crucial for BFSI Compliance)
    # We concatenate First Name, Last Name, and SSN, then hash it with SHA-256
    print(">>> Applying SHA-256 PII Masking...")
    df_masked = df_raw.withColumn(
        "masked_customer_id",
        sha2(concat(col("first_name"), col("last_name"), col("ssn")), 256)
    ).drop("first_name", "last_name", "ssn")  # Drop the raw PII columns

    # 3. Write the clean data back to a new Postgres table
    print(">>> Writing compliant data back to PostgreSQL...")
    df_masked.write.jdbc(
        url=jdbc_url,
        table="clean_customer_data",
        mode="overwrite",
        properties=connection_properties
    )

    print(">>> Final Clean DataFrame:")
    df_masked.show(truncate=False)

    spark.stop()
    print(">>> Pipeline Complete!")


if __name__ == "__main__":
    seed_database()
    run_spark_etl()