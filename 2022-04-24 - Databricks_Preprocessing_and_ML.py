# Databricks notebook source
# install mlflow to track the ML tunning experienments information
%pip install --upgrade mlflow==1.18.0

# COMMAND ----------

#MLflow packages importing
import mlflow
import mlflow.spark
from mlflow.tracking import MlflowClient

# COMMAND ----------

#Data processing
import datetime
import numpy as np

# Preprocessing packages
import pyspark.sql.functions as F 
from pyspark.sql.functions import *
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler,MinMaxScaler
from pyspark.ml import Pipeline

#Model training 
from pyspark.ml.regression import LinearRegression,DecisionTreeRegressor
from pyspark.ml.classification import DecisionTreeClassifier,RandomForestClassifier
from pyspark.ml.evaluation import RegressionEvaluator,MulticlassClassificationEvaluator

# COMMAND ----------

try:
  import mlflow.pyspark.ml
  mlflow.pyspark.ml.autolog()
except:
  print(f"Your version of MLflow ({mlflow.__version__}) does not support pyspark.ml for autologging. To use autologging, upgrade your MLflow client version or use Databricks Runtime for ML 8.3 or above.")

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/Load_from_db.csv"
file_type = "csv"

spark_df=spark.read.csv(file_location,header=True,inferSchema =True)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Data cleaning

# COMMAND ----------

# Convert the pandas df to spark df for data processing
type(spark_df)

# Check the features and datatype in the dataframe
spark_df.printSchema()

# Check missing value
missing_value=spark_df.select([count(when(isnan(c), c)).alias(c) for (c,c_type) in spark_df.dtypes if c_type not in ('timestamp', 'date')])

# indentify the columns with na
na_columns = [ column for column in missing_value.columns if missing_value.agg(F.sum(column)).collect()[0][0]>0 ]

# show columns with na and the number of na 
missing_value.select(na_columns).show()

# Calculate the percentage of na in a column
pct_na=spark_df.select([(count(when(isnan(c), c))/count(lit(1))).alias(c) for (c,c_type) in spark_df.dtypes if c_type not in ('timestamp', 'date')])

# show columns with na and the percentage of na 
pct_na.select(na_columns).show()

# Drop the columns with more than a half of na
spark_df=spark_df.drop("chase_wins","chase_stage_wins")

def mean(df,column):
    return df.select(sum(F.nanvl(F.col(column), F.lit(0)))).collect()[0][0]/len(df.select(column).collect())

# get the mean value of the columns which needs to replace na
replace_na_list=['chase_bonus']
mean_value=[]
for i in replace_na_list:
    mean_value.append(mean(spark_df,i))

# Fill na with mean value
spark_df=spark_df.na.fill({replace_na_list[0]: mean_value[0]})


# COMMAND ----------

# MAGIC %md
# MAGIC ###Feature Engineering

# COMMAND ----------

# Get the age of the driver
spark_df=spark_df.withColumn('birth_year',year("birthday"))
spark_df=spark_df.withColumn('age',2022-spark_df['birth_year'])

# Get the experience in nascar races of the drivers
spark_df=spark_df.withColumn('experience',2022-spark_df['rookie_year'])

# Drop the columns after transformation
spark_df=spark_df.drop("birth_year","rookie_year","birthday")

# DNF ratio (DNF to top 20 score ratio)
spark_df=spark_df.withColumn('DNF_ratio',spark_df['dnf']/spark_df['top_20'])

spark_df=spark_df.na.fill({'DNF_ratio':0})

# Drop the columns after transformation
spark_df=spark_df.drop("dnf","top_20")

# COMMAND ----------

# Transform Dependent variable into 4 classes
spark_df = spark_df.withColumn('rank_class',
    F.when(F.col("rank").between(0, 10), 1)\
    .when(F.col('rank').between(11,30), 2)\
    .when(F.col('rank').between(31,50), 3)\
    .otherwise(4)
)

spark_df=spark_df.drop("rank")

# COMMAND ----------

# MAGIC %md
# MAGIC ###Data exploration

# COMMAND ----------

#Summary statistics
spark_df.select(['rank_class','poles','top_5','chase_bonus','avg_start_position','avg_finish_position',
                 'avg_laps_completed','laps_led_pct','age','experience','DNF_ratio']).describe().show(5,False)


# COMMAND ----------

# check for correlation
non_string_col=['rank_class','poles','top_5','chase_bonus','avg_start_position','avg_finish_position',
                'avg_laps_completed','laps_led_pct','age','experience','DNF_ratio']

for x in non_string_col:
    for y in non_string_col:
        if x!=y:
            if spark_df.select(corr(x,y)).collect()[0][0]>=0.7:
                spark_df.select(corr(x,y)).show()


# COMMAND ----------

# MAGIC %md
# MAGIC ###Preparing Data for Machine Learning

# COMMAND ----------

# Specify independent deatures needed for ml
feature_cols=['series', 'gender', 'country','poles','top_5','chase_bonus','avg_start_position','avg_finish_position',
                'avg_laps_completed','laps_led_pct','age','experience','DNF_ratio']
#Specify the categorical columns
categoricalColumns = ['series', 'gender', 'country']
#Specify the numerical columns
numericCols = ['top_5','chase_bonus','avg_start_position','avg_finish_position','avg_laps_completed',
               'laps_led_pct','age','experience','DNF_ratio']

#Category Indexing and One-Hot Encoding
stages = []
for categoricalCol in categoricalColumns:
    stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
    encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    stages += [stringIndexer, encoder]


# COMMAND ----------

# Iterating over columns to be scaled
for i in numericCols:
    # VectorAssembler Transformation - Converting column to vector type
    assemble_num = VectorAssembler(inputCols=[i],outputCol=i+"_Vect")

    # MinMaxScaler Transformation
    scaler = MinMaxScaler(inputCol=i+"_Vect", outputCol=i+"_Scaled")
    stages += [assemble_num,scaler]

#Merges multiple columns into a vector column
assemblerInputs = [c + "classVec" for c in categoricalColumns] +  [n + "_Scaled" for n in numericCols]
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]


# COMMAND ----------

# MAGIC %md
# MAGIC ###Using pipeline to fit and transform

# COMMAND ----------

# Chain multiple Transformers and Estimators together to specify ml workflow
pipeline = Pipeline(stages = stages)
pipelineModel = pipeline.fit(spark_df)
spark_df = pipelineModel.transform(spark_df)
spark_df.printSchema()

model_df=spark_df.select('features','rank_class')

print((model_df.count(), len(model_df.columns)))


# COMMAND ----------

# MAGIC %md
# MAGIC ###Train test split

# COMMAND ----------

#split the data for train test purpose
train_df,test_df=model_df.randomSplit([0.75,0.25],seed=14)

print((train_df.count(), len(train_df.columns)))
print((test_df.count(), len(test_df.columns)))

# COMMAND ----------

# MAGIC %md
# MAGIC **Linear regression**

# COMMAND ----------

def lr_train(elasticNetParam,loss,solver):
    
    with mlflow.start_run(run_name='Linear_Regression') as run:
    
        #Fit the data
        lr=LinearRegression(featuresCol='features', labelCol='rank_class',elasticNetParam=elasticNetParam,loss=loss,solver=solver)
        model_lr=lr.fit(train_df)
    
        #Log parameters
        mlflow.log_param("elasticNetParam",elasticNetParam)
        mlflow.log_param("loss",loss)
        mlflow.log_param("solver",solver)


        # Log the model for this run
        mlflow.spark.log_model(model_lr,"PySpark-ML-Linear-Regression")

        #Predict
        prediction_lr=model_lr.transform(test_df)

        #Save the prediction
        prediction_lr.toPandas().to_csv("model_lr_predictions.csv",index=False)

        #Log the saved prediction as aerifact
        mlflow.log_artifact("model_lr_predictions.csv")

        #Evaluate the model
        regressionEvaluator = RegressionEvaluator(predictionCol="prediction",labelCol="rank_class")
        rmse=regressionEvaluator.setMetricName("rmse").evaluate(prediction_lr)
        r2=regressionEvaluator.setMetricName("r2").evaluate(prediction_lr)
        mse=regressionEvaluator.setMetricName("mse").evaluate(prediction_lr)
        mae=regressionEvaluator.setMetricName("mae").evaluate(prediction_lr)

        #Log metrics
        mlflow.log_metric("rmse",rmse)
        mlflow.log_metric("r2",r2)
        mlflow.log_metric("mse",mse)
        mlflow.log_metric("mae",mae)
        
        return model_lr, rmse,r2,mse, mae

# COMMAND ----------

model_name, rmse,r2,mse, mae = lr_train(0.8,'squaredError','auto')
print(f"The {model_name} achieved an rmse score of {rmse}, r2 score of {r2}, mse score of {mse}, mae score of {mae} on the validation data")

# COMMAND ----------

# MAGIC %md
# MAGIC **Decision Tree Regressor**

# COMMAND ----------

def dtr_train(maxDepth,minInstancesPerNode,maxBins,impurity):
    
    with mlflow.start_run(run_name='DecisionTree_Regressor') as run:
        
        #Fit the data
        dtr = DecisionTreeRegressor(featuresCol='features', labelCol='rank_class',maxDepth=maxDepth,minInstancesPerNode=minInstancesPerNode,
                                         maxBins=maxBins,impurity=impurity)
        model_dtr=dtr.fit(train_df)

        #Log parameters
        mlflow.log_param("maxDepth",maxDepth)
        mlflow.log_param("minInstancesPerNode",minInstancesPerNode)
        mlflow.log_param("maxBins",maxBins)
        mlflow.log_param("impurity",impurity)


        # Log the model for this run
        mlflow.spark.log_model(model_dtr,"PySpark-ML-DecisionTree-Regressor")

        #Predict
        prediction_dtr=model_dtr.transform(test_df)

        #Save the prediction
        prediction_dtr.toPandas().to_csv("model_dtr_predictions.csv",index=False)

        #Log the saved prediction as aerifact
        mlflow.log_artifact("model_dtr_predictions.csv")

        #Evaluate the model
        regressionEvaluator = RegressionEvaluator(predictionCol="prediction",labelCol="rank_class")
        rmse=regressionEvaluator.setMetricName("rmse").evaluate(prediction_dtr)
        r2=regressionEvaluator.setMetricName("r2").evaluate(prediction_dtr)
        mse=regressionEvaluator.setMetricName("mse").evaluate(prediction_dtr)
        mae=regressionEvaluator.setMetricName("mae").evaluate(prediction_dtr)

        #Log metrics
        mlflow.log_metric("rmse",rmse)
        mlflow.log_metric("r2",r2)
        mlflow.log_metric("mse",mse)
        mlflow.log_metric("mae",mae)
        
        return model_dtr, rmse,r2,mse, mae

# COMMAND ----------

model_name, rmse,r2,mse, mae = dtr_train(8,200,5,'variance')
print(f"The {model_name} achieved an rmse score of {rmse}, r2 score of {r2}, mse score of {mse}, mae score of {mae} on the validation data")

# COMMAND ----------

# MAGIC %md
# MAGIC **Decsion Tree Classfier**

# COMMAND ----------

def dtc_train(maxDepth,minInstancesPerNode,maxBins):
    
    with mlflow.start_run(run_name='DecisionTree_Classifier') as run:
        
        #Fit the data
        dtc = DecisionTreeClassifier(featuresCol='features', labelCol='rank_class',maxDepth=maxDepth,minInstancesPerNode=minInstancesPerNode,
                                         maxBins=maxBins)
        model_dtc=dtc.fit(train_df)

        #Log parameters
        mlflow.log_param("maxDepth",maxDepth)
        mlflow.log_param("minInstancesPerNode",minInstancesPerNode)
        mlflow.log_param("maxBins",maxBins)


        # Log the model for this run
        mlflow.spark.log_model(model_dtc,"PySpark-ML-DecisionTree-Classifier")

        #Predict
        prediction_dtc=model_dtc.transform(test_df)

        #Save the prediction
        prediction_dtc.toPandas().to_csv("model_dtc_predictions.csv",index=False)

        #Log the saved prediction as aerifact
        mlflow.log_artifact("model_dtc_predictions.csv")

        #Evaluate the model
        evaluator = MulticlassClassificationEvaluator(predictionCol="prediction",labelCol="rank_class")
        accuracy=evaluator.setMetricName("accuracy").evaluate(prediction_dtc)
        f1=evaluator.setMetricName("f1").evaluate(prediction_dtc)
        recall=evaluator.setMetricName("weightedRecall").evaluate(prediction_dtc)
        precision=evaluator.setMetricName("weightedPrecision").evaluate(prediction_dtc)

        #Log metrics
        mlflow.log_metric("accuracy",accuracy)
        mlflow.log_metric("f1",f1)
        mlflow.log_metric("recall",recall)
        mlflow.log_metric("precision",precision)
        
        return model_dtc, accuracy,f1,recall, precision

# COMMAND ----------

model_name, accuracy,f1,recall,precision = dtc_train(8,200,6)
print(f"The {model_name} achieved an accuracy score of {accuracy}, F1 score of {f1}, recall score of {recall}, precision score of {precision} on the validation data")

# COMMAND ----------

# MAGIC %md
# MAGIC **Random Forest Classfier**

# COMMAND ----------

def rfc_train(numTrees,maxDepth,maxBins):
    
    with mlflow.start_run(nested=True,run_name='RandomForest_Classifier') as run:

        #Fit the data
        rfc = RandomForestClassifier(featuresCol='features', labelCol='rank_class',maxDepth=maxDepth,numTrees=numTrees,
                                         maxBins=maxBins)
        model_rfc=rfc.fit(train_df)

        #Log parameters
        mlflow.log_param("maxDepth",maxDepth)
        mlflow.log_param("numTrees",numTrees)
        mlflow.log_param("maxBins",maxBins)


        # Log the model for this run
        mlflow.spark.log_model(model_rfc,"PySpark-ML-RandomForest_Classifier")

        #Predict
        prediction_rfc=model_rfc.transform(test_df)

        #Save the prediction
        prediction_rfc.toPandas().to_csv("model_rfc_predictions.csv",index=False)

        #Log the saved prediction as aerifact
        mlflow.log_artifact("model_rfc_predictions.csv")

        #Evaluate the model
        evaluator = MulticlassClassificationEvaluator(predictionCol="prediction",labelCol="rank_class")
        accuracy=evaluator.setMetricName("accuracy").evaluate(prediction_rfc)
        f1=evaluator.setMetricName("f1").evaluate(prediction_rfc)
        recall=evaluator.setMetricName("weightedRecall").evaluate(prediction_rfc)
        precision=evaluator.setMetricName("weightedPrecision").evaluate(prediction_rfc)

        #Log metrics
        mlflow.log_metric("accuracy",accuracy)
        mlflow.log_metric("f1",f1)
        mlflow.log_metric("recall",recall)
        mlflow.log_metric("precision",precision)
        
        return model_rfc, accuracy,f1,recall, precision

# COMMAND ----------

model_name, accuracy,f1,recall,precision = rfc_train(200,8,4)
print(f"The {model_name} achieved an accuracy score of {accuracy}, F1 score of {f1}, recall score of {recall}, precision score of {precision} on the validation data")

# COMMAND ----------

# MAGIC %md
# MAGIC ###Hyperparameter Tuning

# COMMAND ----------

! pip install hyperopt
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

# COMMAND ----------

def train_with_hyperopt(params):
    numTrees = int(params['numTrees'])
    maxDepth = int(params['maxDepth'])
    maxBins = int(params['maxBins'])
    model_name, accuracy,f1,recall,precision = rfc_train(numTrees,maxDepth,maxBins)
    loss = - accuracy
    return {'loss': loss, 'status': STATUS_OK}

import numpy as np
space = {
  'numTrees': hp.uniform('numTrees', 50, 500),
  'maxBins': hp.uniform('maxBins', 2, 32),
  'maxDepth': hp.uniform('maxDepth', 2, 20)
}
algo=tpe.suggest

# COMMAND ----------

with mlflow.start_run():
    best_params = fmin(
    fn=train_with_hyperopt,
    space=space,
    algo=algo,
    max_evals=8)
    
    best_numTrees = int(best_params['numTrees'])
    best_maxDepth = int(best_params['maxDepth'])
    best_maxBins = int(best_params['maxBins'])
    
    #Log parameters
    mlflow.log_param("maxDepth",best_maxDepth)
    mlflow.log_param("numTrees",best_numTrees)
    mlflow.log_param("maxBins",best_maxBins)
    
    
    final_model, accuracy, f1, recall, precision = rfc_train(best_numTrees, best_maxDepth,best_maxBins)
    
    #Log metrics
    mlflow.log_metric("accuracy",accuracy)
    mlflow.log_metric("f1",f1)
    mlflow.log_metric("recall",recall)
    mlflow.log_metric("precision",precision)
 

# COMMAND ----------

print(f"On the test data, the tuned model {final_model} achieved an accuracy score of {accuracy}, F1 score of {f1}, recall score of {recall}, precision score of {precision} on the validation data")

# COMMAND ----------

# MAGIC %md
# MAGIC ###Model deployment

# COMMAND ----------

logged_model = 'runs:/6f092cd0971d45e8a8ebfaf01629b6ce/PySpark-ML-RandomForest_Classifier'
final_model = mlflow.spark.load_model(logged_model)

# COMMAND ----------

# Read input data for prediction
file_location = "/FileStore/tables/input_new_data.csv"
file_type = "csv"

spark_data=spark.read.csv(file_location,header=True,inferSchema =True)

# COMMAND ----------

def tranform_input_data(data):
    data_transformed = pipelineModel.transform(data)
    new=data_transformed.select('features')
    return new

# COMMAND ----------

new=tranform_input_data(spark_data)

# COMMAND ----------

deployment = final_model.transform(new)
deployment.show()

# COMMAND ----------

#add a model to the registry
result = mlflow.register_model(logged_model,
    "PySpark-ML-RandomForest_Classifier"
)

# COMMAND ----------

# MAGIC %md
# MAGIC **Call the model through API**

# COMMAND ----------

import os
import requests
import numpy as np
import pandas as pd

def create_tf_serving_json(data):
    return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
    url = 'https://dbc-a737d754-7526.cloud.databricks.com/model/PySpark-ML-RandomForest_Classifier/1/invocations'
    headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}'}
    data_json = dataset.to_dict(orient='split') if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
    response = requests.request(method='POST', headers=headers, url=url, json=data_json)
    if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
    return response.json()
