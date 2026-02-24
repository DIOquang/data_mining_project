from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count

def main():
    spark = SparkSession.builder.getOrCreate()

    ACCESS_KEY = ""
    SECRET_KEY = ""
    BUCKET_NAME = "books-s3-demo"
    
    spark.conf.set("fs.s3a.access.key", ACCESS_KEY)
    spark.conf.set("fs.s3a.secret.key", SECRET_KEY)
    spark.conf.set("fs.s3a.endpoint", "s3.amazonaws.com")

    print("Đang tải dữ liệu Books từ S3 vào Databricks...")
    file_path = f"s3a://books-s3-demo/Books.jsonl.gz"
    df_raw = spark.read.json(file_path)

    df_selected = df_raw.select(
        col("user_id"),
        col("parent_asin").alias("item_id"),
        col("rating"),
        col("timestamp")
    ).dropna(subset=["user_id", "item_id", "rating"])
    
    item_counts = df_selected.groupBy("item_id").agg(count("*").alias("i_count"))
    popular_items = item_counts.filter(col("i_count") >= 5)
    df_step1 = df_selected.join(popular_items, "item_id", "inner").drop("i_count")

    user_counts = df_step1.groupBy("user_id").agg(count("*").alias("u_count"))
    active_users = user_counts.filter(col("u_count") >= 5)
    df_clean = df_step1.join(active_users, "user_id", "inner").drop("u_count")

    output_path = f"s3a://{BUCKET_NAME}/processed/Books_Reviews_Clean.parquet"
    df_clean.write.mode("overwrite").parquet(output_path)
    print("fdsfsdf")
if __name__ == "__main__":
    main()
