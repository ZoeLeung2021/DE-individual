# from airflow import DAG
from marquez_airflow import DAG
from airflow.models import BaseOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.postgres_operator import PostgresOperator
from airflow.operators.python_operator import PythonOperator
from airflow.hooks.S3_hook import S3Hook
from airflow.models import Variable

from airflow.hooks.postgres_hook import PostgresHook

from datetime import datetime
from datetime import timedelta
import logging
import pandas as pd


log = logging.getLogger(__name__)


# =============================================================================
# 1. Set up the main configurations of the dag
# =============================================================================

default_args = {
    'start_date': datetime(2022, 2, 11),
    'owner': 'Airflow',
    'filestore_base': '/tmp/airflowtemp/',
    'bucket_name':'de-individual',
    'prefix': 'extra_folder',
    'db_name': Variable.get("schema_postgres",deserialize_json=True)['db_name'],
    'aws_conn_id':"aws_default",
    'postgres_conn_id': 'postgres_conn_id',
    'email_on_failure': False,
    'email_on_retry': False,
    # 'retries': 1,
    'retries': 0,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG('marquez_postgres',
          description='Test postgres operations',
          schedule_interval='@weekly',
          catchup=False,
          default_args=default_args,
          max_active_runs=1)


# =============================================================================
# 2. Define different functions
# =============================================================================

def create_rest_tables_in_db(**kwargs):

    log.info('received:{0}'.format(kwargs))

    log.info('default arguments received:{0}'.format(kwargs))
    print('default arguments',kwargs)

    task_instance = kwargs['ti']

    schema_postgres = Variable.get("schema_postgres",deserialize_json=True)
    log.info('schema_postgres dictionary values:{0}'.format(schema_postgres))

    bucket_name = kwargs['bucket_name']
    key=schema_postgres['key']

    s3=S3Hook(kwargs['aws_conn_id'])

    print("Reading SQL File with queries to create database")
    sql_queries=s3.read_key(key,bucket_name)


    #connect to the database
    pg_hook=PostgresHook(postgres_conn_id=kwargs["postgres_conn_id"],schema=kwargs['db_name'])
    connection=pg_hook.get_conn()
    cursor=connection.cursor()

    print("Executing sql queries")
    cursor.execute(sql_queries)


def query_save_result(**kwargs):

    import pandas as pd
    import io

    #connect to the database
    pg_hook=PostgresHook(postgres_conn_id=kwargs["postgres_conn_id"],schema=kwargs['db_name'])

    task_instance=kwargs['ti']

    print(task_instance)

    log.info('received:{0}'.format(kwargs))
    log.info('default arguments received:{0}'.format(kwargs))
    
    save_csv=Variable.get("save_csv",deserialize_json=True)
    log.info('save_csv dictionary values:{0}'.format(save_csv))

    bucket_name=kwargs['bucket_name']
    key=save_csv['key']
    s3=S3Hook(kwargs['aws_conn_id'])

    sql_query="""
              select * from nascar.personal_info
              """
    
    #Execute query and save result to pandas dataframe 
    print("Executing query")
    df=pg_hook.get_pandas_df(sql_query)
    print(df)

    #Prepare the file to send to s3
    csv_buffer=io.StringIO()
    df.to_csv(csv_buffer,index=False)

    #save the pandas dataframe as a csv to s3
    s3=s3.get_resource_type('s3')

    #Get the data type object from pandas dataframe, key and connection object to s3 bucket
    data = csv_buffer.getvalue()

    print("Saving CSV file")
    object = s3.Object(bucket_name,key)

    #Write the file to s3 bucket in specific path
    object.put(Body=data)


import boto3
import csv
    
def insert_personal_info_data_func (**kwargs):
    log.info('received:{0}'.format(kwargs))
    log.info('default arguments received:{0}'.format(kwargs))
    print('default arguments',kwargs)
    task_instance = kwargs['ti']
    personal_info_data = Variable.get("personal_info_data",deserialize_json=True)
    log.info('personal_info_data dictionary values:{0}'.format(personal_info_data))
    bucket_name = kwargs['bucket_name']
    key=personal_info_data['key']
    s3=S3Hook(kwargs['aws_conn_id'])
    print("Reading data from s3")
    s3=s3.get_resource_type('s3')
    resp = s3.Object(bucket_name,key).get()
    data = resp['Body'].read().decode('utf-8')
    data=data.split("\n")


    pg_hook=PostgresHook(postgres_conn_id=kwargs["postgres_conn_id"],schema=kwargs['db_name'])
    connection=pg_hook.get_conn()
    cursor=connection.cursor()

    for row in data:
        try:
            row=row.replace("\n","").split(",")
            for i in range(1, len(row)):
                cursor.execute('insert into personal_info values(str(row)[i])')
                connection.commit()
        except:
            continue
        print(str(row)[1])
    print(cursor.rowcount,"insert successfully")





# =============================================================================
# 3. Set up the dags
# =============================================================================

validate_table_exists = PostgresOperator(
    task_id='validate_table_exists',
    postgres_conn_id='postgres_conn_id',
    sql='''
            drop table if exists nascar.personal_info CASCADE;
            create table nascar.personal_info (
                id                      serial primary key,
                driver_id               varchar(256),
                first_name              varchar(256),
                last_name               varchar(256),
                full_name               varchar(256),
                gender                  varchar(256),
                height                  float,
                weight                  float,
                birthday                timestamp,
                birth_place             varchar(256),
                country                 varchar(256)
                ); 
            ''',
                dag=dag
                )

create_rest_tables=PythonOperator(task_id='create_rest_tables_in_db',
                          python_callable=create_rest_tables_in_db,
                          op_kwargs=default_args,
                          provide_context=True,
                          dag=dag)



insert_data=PythonOperator(task_id='insert_data',
                          python_callable=insert_personal_info_data_func,
                          op_kwargs=default_args,
                          provide_context=True,
                          dag=dag)


query_save_csv=PythonOperator(task_id='query_save_csv',
                       python_callable=query_save_result,
                       op_kwargs=default_args,
                       provide_context=True,
                       dag=dag)

# =============================================================================
# 4. Indicating the order of the dags
# =============================================================================

validate_table_exists >> create_rest_tables >> insert_data >> query_save_csv