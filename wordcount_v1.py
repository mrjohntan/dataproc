#! /usr/bin/python
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql import functions as F
from pyspark.conf import SparkConf
import pyspark
from string import punctuation

def process(col):
    col = F.lower(col)                      # convert to lowercase
    col = F.translate(col, punctuation, '') # remove punctuation
    col = F.trim(col)                       # remove leading and traling blank space
    col = F.split(col, '\s')                # split on blank space
    col = F.explode(col)                    # give each iterable in row its owwn row
    return col

spark = SparkSession.builder \
    .master("yarn") \
    .appName("Word Count") \
    .config("spark.shuffle.service.enabled", "true") \
    .config("spark.dynamicAllocation.enabled", "true") \
    .config("spark.dynamicAllocation.minExecutors", "0") \
    .config("spark.dynamicAllocation.maxExecutors", "100") \
    .getOrCreate()

# sc = SparkContext()
# spark = SparkSession(sc)
# spark.conf.set("spark.dynamicAllocation.minExecutors", 3)
# spark.conf.set("spark.dynamicAllocation.maxExecutors", 5)
df = spark.read.json("gs://dp2-testdata-9902/test_review.json")
df2 = df.select("text").filter(df['text'] != '')
words = df2.withColumn('text', process(df2.text)).filter(df2['text'] != '')
counts = words.groupby('text').count()
counts = counts.sort(counts['count'].desc())
counts.show()
counts.coalesce(1).write.mode('overwrite').csv("gs://dp2-testdata-9902/count.csv")