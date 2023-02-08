# Databricks notebook source
# MAGIC %md
# MAGIC # Infer with Deployed Model

# COMMAND ----------

# MAGIC %md
# MAGIC ### Connect to Wallaroo
# MAGIC 
# MAGIC Connect to the Wallaroo instance and set the `housepricing` workspace as the current workspace.

# COMMAND ----------

import json
import pickle
import wallaroo
import pandas as pd
import numpy as np

import simdb # module for the purpose of this demo to simulate pulling data from a database

from wallaroo_client import get_workspace

# COMMAND ----------

# Login to local Wallaroo instance
wallarooPrefix = "beautiful-platypus-3587"
wallarooSuffix = "wallaroo.community"

# wallarooPrefix = "squishy-wallaroo-6187"
# wallarooSuffix = "wallaroo.dev"
 
wl = wallaroo.Client(api_endpoint=f"https://{wallarooPrefix}.api.{wallarooSuffix}", 
                    auth_endpoint=f"https://{wallarooPrefix}.keycloak.{wallarooSuffix}", 
                    auth_type="sso")

# COMMAND ----------

new_workspace = get_workspace(wl, "housepricing")
_ = wl.set_current_workspace(new_workspace)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Deploy the Pipeline
# MAGIC 
# MAGIC Fetch and re-deploy the `housing-pipe` pipeline created in the previous notebook.

# COMMAND ----------

pipeline = wl.pipelines_by_name("housing-pipe")[0]
pipeline.deploy()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read In New House Listings
# MAGIC 
# MAGIC From the data store, load the previous month's house listing and submit them to the deployed pipeline.

# COMMAND ----------

conn = simdb.simulate_db_connection()

# create the query
query = f"select * from {simdb.tablename} where date > DATE(DATE(), '-1 month') AND sale_price is NULL"
print(query)

# read in the data
newbatch = pd.read_sql_query(query, conn)
newbatch.shape

# COMMAND ----------

query = {'query': newbatch.to_json()}
result = pipeline.infer(query)[0]

# COMMAND ----------

predicted_prices = result.data()[0]

# COMMAND ----------

len(predicted_prices)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Quick visualization of results
# MAGIC 
# MAGIC Create a table that links the house listing id to the predicted price

# COMMAND ----------

result_table = pd.DataFrame({
    'id': newbatch['id'],
    'saleprice_estimate': predicted_prices,
})

result_table

# COMMAND ----------

conn.close()
pipeline.undeploy()
