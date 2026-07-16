# Configure the necessary Spark environment
import os
import sys

# Path for java
os.environ['JAVA_HOME'] = "/usr/lib/jvm/java-11-openjdk-amd64"

spark_home = "/opt/spark/"
sys.path.insert(0, spark_home + "/python")

# Add the py4j to the path.
sys.path.insert(0, spark_home + "/python/lib/py4j-0.10.7-src.zip")

# Path for spark source folder
os.environ['SPARK_HOME'] = spark_home

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from pyspark import SparkContext, SparkConf, SQLContext

conf = SparkConf()

conf.setMaster("local[8]").setAppName("Reddit Spark App")
#conf.setMaster("spark://igea:7077").setAppName("Test app")
conf.set("spark.executor.memory", "4g")
conf.set("spark.driver.memory", "6g")
conf.set("spark.ui.port", "4832")
conf.set('spark.driver.maxResultSize','12g')
conf.set("spark.local.dir", "/data/big/tmp/")
conf.set("spark.hadoop.io.compression.codecs", "org.apache.hadoop.io.compress.SnappyCodec")

def spark_context():
     return SparkContext(conf=conf)
    
def sql_context(sc):
     return SQLContext(sc)
