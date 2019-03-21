# Run using the following command
# spark-submit   --executor-cores 7 --executor-memory 10G --num-executors 3 --master yarn classification_cv.py



from pyspark.sql import SparkSession
from pprint import pprint
import time
import os
import sys
os.environ['PYSPARK_PYTHON']='/usr/bin/python3'
os.environ['PYSPARK_DRIVER_PYTHON']='/usr/bin/python3'
spark = SparkSession.builder.appName('Classification on FireService with CV on AWS').getOrCreate()
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

# test train split
print(bcolors.OKBLUE + bcolors.BOLD + 'Creating and Caching Test Train Split' +  bcolors.ENDC)
train, test = df.randomSplit([0.8,0.2])

train.repartition(7).createOrReplaceTempView('trainVIEW')
spark.catalog.cacheTable('trainVIEW')
spark.table("trainVIEW").count() # materialize the view
train = spark.table("trainVIEW")

test.repartition(7).createOrReplaceTempView('testVIEW')
spark.catalog.cacheTable('testVIEW')
spark.table("testVIEW").count() # materialize the view
test = spark.table("testVIEW")

# Classification Evaluator
print(bcolors.OKBLUE + bcolors.BOLD + 'Creating Evaluator' +  bcolors.ENDC)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(labelCol='FinalPriority')
pprint(evaluator.extractParamMap())

# Logistic Regression - cross validation
print(bcolors.OKBLUE + bcolors.BOLD + 'Starting Logistic Regression CV with 3x3x3 parameters' +  bcolors.ENDC)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
start = time.time()
lr = LogisticRegression(featuresCol='features', labelCol='FinalPriority')
# Create ParamGrid for Cross Validation
paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.01, 0.5, 2.0])
             .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
             .addGrid(lr.maxIter, [1, 5, 10])
             .build())
cv = CrossValidator(estimator=lr, 
                    estimatorParamMaps=paramGrid, 
                    evaluator=evaluator, 
                    numFolds=3)
# Run cross validations
cv.setParallelism(7)
cvModel = cv.fit(train)
print('Time for training : {} sec'.format(time.time()-start))

# predict and evaludate
predictions = cvModel.transform(test)
auc = evaluator.evaluate(predictions)
print('Auc : {}'.format(auc))
cvModel.write().overwrite().save('./LogisticRegression_cv.model')

