# Run using the following command
# spark-submit --driver-memory 4G --executor-cores 6 --executor-memory 4G --packages org.postgresql:postgresql:42.1.1,org.apache.hadoop:hadoop-aws:2.7.1,com.datastax.spark:spark-cassandra-connector_2.11:2.3.0 --master local[6] ETL_aws.py


from pyspark.sql import SparkSession
from pprint import pprint
import os
import sys

os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/bin/python3"
spark = SparkSession.builder.appName("FireService ETL").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


from pyspark.sql.functions import col, sum


def missing_df(df):
    return df.select(*(sum(col(c).isNull().cast("int")).alias(c) for c in df.columns))


# Load Data Frame using parquet for faster access
print(bcolors.OKBLUE + bcolors.BOLD + "Reading File" + bcolors.ENDC)
fireServiceDF = spark.read.parquet("s3a://spark23march2019/fire_dept_calls.parquet")



# drop columns which we dont want are having lots of missing values
fireServiceDF = fireServiceDF.drop(
    "Address",
    "AvailableDtTm",
    "Box",
    "CallNumber",
    "CallTypeGroup",
    "HospitalDtTm",
    "IncidentNumber",
    "OnSceneDtTm",
    "OriginalPriority",
    "Priority",
    "ResponseDtTm",
    "RowID",
    "TransportDtTm",
    "UnitID",
    "Unitsequenceincalldispatch",
    "ZipcodeofIncident",
)


numerical_variables = ["FirePreventionDistrict", "NumberofAlarms", "SupervisorDistrict"]
categorical_variables = [
    "ALSUnit",
    "Battalion",
    "CallFinalDisposition",
    "CallType",
    "City",
    "NeighborhoodDistrict",
    "StationArea",
    "UnitType",
]

# stages holds all the transformations

stages = []


# convert the numerical columns
print(bcolors.OKBLUE + bcolors.BOLD + "Converting numerical variables" + bcolors.ENDC)
from pyspark.sql.functions import col


for var in numerical_variables:
    fireServiceDF = fireServiceDF.withColumn(var, col(var).astype("int"))

# fill null values as -1 for numerical variables
print(bcolors.OKBLUE + bcolors.BOLD + "Filling Null Values" + bcolors.ENDC)
fireServiceDF = fireServiceDF.fillna(-1, subset=numerical_variables)

# convert dates from String to Date format
print(bcolors.OKBLUE + bcolors.BOLD + "Converting Date" + bcolors.ENDC)
from pyspark.sql.functions import to_timestamp, to_date, unix_timestamp

from_pattern1 = "MM/dd/yyyy"  # to_pattern1 = 'yyyy-MM-dd'
from_pattern2 = "MM/dd/yyyy hh:mm:ss aa"  # to_pattern2 = 'MM/dd/yyyy hh:mm:ss aa'
fireServiceDF = (
    fireServiceDF.withColumn("CallDate", to_date("CallDate", format=from_pattern1))
    .withColumn("WatchDate", to_date("WatchDate", format=from_pattern1))
    .withColumn("ReceivedDtTm", to_timestamp("ReceivedDtTm", format=from_pattern2))
    .withColumn("DispatchDtTm", to_timestamp("DispatchDtTm", format=from_pattern2))
    .withColumn("EntryDtTm", to_timestamp("EntryDtTm", format=from_pattern2))
)

# String Indexing - convert categorical to numbers and one hot
print(bcolors.OKBLUE + bcolors.BOLD + "Converting String Indexing" + bcolors.ENDC)
from pyspark.ml.feature import (
    OneHotEncoderEstimator,
    StringIndexer,
    VectorAssembler,
    OneHotEncoder,
)

fireServiceDF = fireServiceDF.withColumn("ALSUnit", col("ALSUnit").astype("string"))
for var in categorical_variables:
    indexer = StringIndexer(
        inputCol=var,
        outputCol=var + "_Index",
        handleInvalid="keep",
        stringOrderType="alphabetAsc",
    )
    encoder = OneHotEncoder(
        inputCol=indexer.getOutputCol(), outputCol=var + "_classVec"
    )
    stages += [indexer, encoder]

# create 'features' column using VectorAssembler
from pyspark.ml.feature import VectorAssembler
assemblerInputs = [c + "_classVec" for c in categorical_variables] + numerical_variables
print(bcolors.WARNING + str(assemblerInputs) + bcolors.ENDC)
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]

# create pipeline from stages and apply
print(bcolors.OKBLUE + bcolors.BOLD + "Applying Pipeline" + bcolors.ENDC)
from pyspark.ml import Pipeline
import time
start = time.time()
pipeline_fit = Pipeline(stages=stages).fit(fireServiceDF)
transformed_fireServiceDF = pipeline_fit.transform(fireServiceDF)
pipeline_fit.write().overwrite().save("pipeline.model")
print("Time : {} sec".format(time.time() - start))


# save the transformed data frame to parquet
print(bcolors.OKBLUE + bcolors.BOLD + "Saving Data to Parquet" + bcolors.ENDC)
transformed_fireServiceDF = transformed_fireServiceDF.select(
    assemblerInputs + ["features", "FinalPriority"]
)
transformed_fireServiceDF.printSchema()
transformed_fireServiceDF.write.parquet(
    "./fireservice_data_for_ML2.parquet", mode="overwrite"
)
