from pyspark.sql import SparkSession


def main():
    print(">>> Initializing Spark Session...")

    # Initialize Spark in local mode (runs entirely inside your pyspark-dev container)
    spark = SparkSession.builder \
        .appName("PyCharm_Test") \
        .master("local[*]") \
        .getOrCreate()

    print(">>> Spark Session created successfully!")

    # Create a simple test DataFrame
    data = [("Alice", 34), ("Bob", 45), ("Charlie", 29)]
    columns = ["Name", "Age"]
    df = spark.createDataFrame(data, columns)

    print(">>> Showing DataFrame:")
    df.show()

    spark.stop()
    print(">>> Test Complete!")


if __name__ == "__main__":
    main()