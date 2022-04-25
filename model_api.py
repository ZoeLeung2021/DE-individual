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