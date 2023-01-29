from pyspark.sql.session import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.functions import udf
from pyspark.ml.feature import * 
from nltk.stem.porter import *
import string 
import re

spark = SparkSession.builder \
    .master("yarn") \
    .appName("wordcount") \
    .config("spark.shuffle.service.enabled", "true") \
    .config("spark.dynamicAllocation.enabled", "true") \
    .config("spark.dynamicAllocation.minExecutors", "0") \
    .config("spark.dynamicAllocation.maxExecutors", "25") \
    .getOrCreate()

review = spark.read.json("gs://dp2-testdata-9902/test_review.json")
# review.show(5)

# cache  dataframes
# review.cache()

# remove punctuation
def remove_punct(text):
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
    nopunct = regex.sub(" ", text) 
    return nopunct

# udf
punct_remover = udf(lambda x: remove_punct(x))

# apply to review raw data
review_df = review.select('review_id', punct_remover('text'))
review_df = review_df.withColumnRenamed('<lambda>(text)', 'text')

# review_df.show(5)

# tokenize
tok = Tokenizer(inputCol="text", outputCol="words")
review_tokenized = tok.transform(review_df)

# remove stop words
stopword_rm = StopWordsRemover(inputCol='words', outputCol='words_nsw')
review_tokenized = stopword_rm.transform(review_tokenized)
# review_tokenized.show(5)

# create word list
wordlist = review_tokenized.select("words_nsw").select(explode("words_nsw").alias("word"))
wordlist = wordlist.filter(wordlist.word != "")
# wordlist.show(5)

# generate word count
wordcount = wordlist.groupby("word").count()
wordcount = wordcount.sort(wordcount['count'].desc())
wordcount.show()

# saving wordcount output
wordcount.coalesce(1).write.mode('overwrite').csv("gs://dp2-testdata-9902/wordcount.csv")

spark.stop()