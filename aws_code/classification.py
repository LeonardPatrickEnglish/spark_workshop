# Run using the following command
# spark-submit  --driver-memory 4G --executor-cores 6 --executor-memory 8G --packages org.postgresql:postgresql:42.1.1,org.apache.hadoop:hadoop-aws:2.7.1,com.datastax.spark:spark-cassandra-connector_2.11:2.3.0 --master local[6] classification.py



from pyspark.sql import SparkSession
from pprint import pprint
import time
import os
import sys
os.environ['PYSPARK_PYTHON']='/usr/bin/python3'
os.environ['PYSPARK_DRIVER_PYTHON']='/usr/bin/python3'
spark = SparkSession.builder.appName('Classification on FireService without CV').getOrCreate()
spark.sparkContext.setLogLevel('ERROR')

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# load data from parquet
print(bcolors.OKBLUE + bcolors.BOLD + 'Reading File' +  bcolors.ENDC)
df = spark.read.parquet('./fireservice_data_for_ML2.parquet')
df = df.select('features', 'FinalPriority')
df.cache()
df.show(5)


# test train split
print(bcolors.OKBLUE + bcolors.BOLD + 'Creating Test Train Split' +  bcolors.ENDC)
train, test = df.randomSplit([0.8,0.2])

# Classification Evaluator
print(bcolors.OKBLUE + bcolors.BOLD + 'Creating Evaluator' +  bcolors.ENDC)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from sklearn.metrics import accuracy_score
evaluator = BinaryClassificationEvaluator(labelCol='FinalPriority')
pprint(evaluator.extractParamMap())

# Logistic Regression - cross validation
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
start = time.time()
print(bcolors.OKBLUE + bcolors.BOLD + 'Starting Logistic Regression without CV' +  bcolors.ENDC)
lr = LogisticRegression(featuresCol='features', labelCol='FinalPriority')
cvModel = lr.fit(train)
print('Time for training : {} sec'.format(time.time()-start))
# predict and evaludate
predictions = cvModel.transform(test)
auc = evaluator.evaluate(predictions)
print('Auc : {}'.format(auc))
cvModel.write().overwrite().save('./LogisticRegression.model')

