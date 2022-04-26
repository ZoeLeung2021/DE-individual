# DE-individual

**1. Data collection**

1.1 Jupyter notebook: 
- API_with_func.ipynb
- Web_scrpaing.ipynb
- Data_collection.ipynb (Combing the above two notebook, written with comment)

1.2 CSV files:
- collection_output (2 main data frame: driver_stats & driver_standings)
- other_csv (intermediate output during the collection stage)

**2. Data storage**
2.1 airflow folder
2.2 schema.sql
2.3 data_storage.ipynb

**3.ML**
3.1 Notebook (the below two are 90% same except the specify ones)
- Databricks_Preprocessing_and_ML_with_UI_tracking (with UI interface screenshot)
- Databricks_Preprocessing_and_ML_pipeline (with Deployment)

3.2 Pyhton script:
- model_api.py (serve with Model API)

3.3 Machine_Learning_data folder (csv data)

**4.Set up**

4.1 terraform
- /github/workflow
- main.tf

4.2 Spark installatiom
- install_spark.sh
- spark-3.2.1-bin-hadoop3.2 folder

4.3 Postgres
- postgresql-42.3.2
