# Databricks notebook source
# MAGIC %md
# MAGIC # Training Notebook
# MAGIC 
# MAGIC We will train an xgboost model to predict house sale prices in this housing market.
# MAGIC 
# MAGIC ### Retrieve Training Data
# MAGIC 
# MAGIC Note that this connection is simulated to demonstrate how data would be retrieved from an existing data store.  For training, we will use the data on all houses sold in this market with the last two years.

# COMMAND ----------

import numpy as np
import pandas as pd

import sklearn

import xgboost as xgb

import seaborn
import matplotlib
import matplotlib.pyplot as plt

import pickle

import simdb # module for the purpose of this demo to simulate pulling data from a database

from preprocess import create_features  # our custom preprocessing
from postprocess import postprocess    # our custom postprocessing

matplotlib.rcParams["figure.figsize"] = (12,6)

# COMMAND ----------

conn = simdb.simulate_db_connection()
tablename = simdb.tablename

query = f"select * from {tablename} where date > DATE(DATE(), '-24 month') AND sale_price is not NULL"
print(query)
# read in the data
housing_data = pd.read_sql_query(query, conn)

conn.close()
housing_data

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data transformations
# MAGIC 
# MAGIC To improve relative error performance, we will predict on `log10` of the sale price.
# MAGIC 
# MAGIC Predict on log10 price to try to improve relative error performance

# COMMAND ----------

housing_data['logprice'] = np.log10(housing_data.list_price)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Split Data into train/test/validation

# COMMAND ----------

# split data into training and test
outcome = 'logprice'

runif = np.random.default_rng(2206222).uniform(0, 1, housing_data.shape[0])
gp = np.where(runif < 0.2, 'test', 'training')

hd_train = housing_data.loc[gp=='training', :].reset_index(drop=True, inplace=False)
hd_test = housing_data.loc[gp=='test', :].reset_index(drop=True, inplace=False)

# split the training into training and val for xgboost
runif = np.random.default_rng(123).uniform(0, 1, hd_train.shape[0])
xgb_gp = np.where(runif < 0.2, 'val', 'train')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Preprocess the data (create the features we will use in the model) and train.

# COMMAND ----------

# for xgboost
train_features = hd_train.loc[xgb_gp=='train', :].reset_index(drop=True, inplace=False)
train_features = np.array(create_features(train_features))  
train_labels = np.array(hd_train.loc[xgb_gp=='train', outcome])

val_features = hd_train.loc[xgb_gp=='val', :].reset_index(drop=True, inplace=False)
val_features = np.array(create_features(val_features))
val_labels = np.array(hd_train.loc[xgb_gp=='val', outcome])

print(f'train_features: {train_features.shape}, train_labels: {len(train_labels)}')
print(f'val_features: {val_features.shape}, val_labels: {len(val_labels)}')


# COMMAND ----------

# MAGIC %md
# MAGIC ### Generate and Test the Model
# MAGIC 
# MAGIC Based on the experimentation and testing performed in a previous stage,  XGBoost was selected as the ML model and the variables for training were selected.  The model will be generated and tested against sample data.

# COMMAND ----------


xgb_model = xgb.XGBRegressor(
    objective = 'reg:squarederror', 
    max_depth=5, 
    base_score = np.mean(hd_train[outcome])
    )

xgb_model.fit( 
    train_features,
    train_labels,
    eval_set=[(train_features, train_labels), (val_features, val_labels)],
    verbose=False,
    early_stopping_rounds=35
)


# COMMAND ----------

test_features = np.array(create_features(hd_test.copy()))
test_labels = np.array(hd_test.loc[:, outcome])

pframe = pd.DataFrame({
    'pred' : postprocess(xgb_model.predict(test_features)),
    'actual' : postprocess(test_labels)
})

ax = seaborn.scatterplot(
    data=pframe,
    x='pred',
    y='actual',
    alpha=0.2
)
matplotlib.pyplot.plot(pframe.pred, pframe.pred, color='DarkGreen')
matplotlib.pyplot.title("test")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Get hold-out performance**

# COMMAND ----------

pframe['se'] = (pframe.pred - pframe.actual)**2

pframe['pct_err'] = 100*np.abs(pframe.pred - pframe.actual)/pframe.actual

# COMMAND ----------

rmse = np.sqrt(np.mean(pframe.se))
mape = np.mean(pframe.pct_err)

print(f'rmse = {rmse}, mape = {mape}')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Convert the Model to Onnx
# MAGIC 
# MAGIC This step converts the model to onnx for easy import into Wallaroo.

# COMMAND ----------

import onnx
from onnxmltools.convert import convert_xgboost

from skl2onnx.common.data_types import FloatTensorType, DoubleTensorType

import preprocess

# set the number of columns
ncols = len(preprocess._vars)

# derive the opset value

from onnx.defs import onnx_opset_version
from onnxconverter_common.onnx_ex import DEFAULT_OPSET_NUMBER
TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())

# COMMAND ----------

# Convert the model to onnx

onnx_model_converted = convert_xgboost(xgb_model, 'tree-based classifier',
                             [('input', FloatTensorType([None, ncols]))],
                             target_opset=TARGET_OPSET)

# Save the model
onnx.save_model(onnx_model_converted, "housing_model_xgb.onnx")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Deploy model
# MAGIC 
# MAGIC ### Connect to Wallaroo

# COMMAND ----------

import json
import wallaroo

# COMMAND ----------

def get_workspace(name):
    workspace = None
    for ws in wl.list_workspaces():
        if ws.name() == name:
            workspace= ws
    if(workspace == None):
        workspace = wl.create_workspace(name)
    return workspace

def get_pipeline(name):
    try:
        pipeline = wl.pipelines_by_name(pipeline_name)[0]
    except EntityNotFoundError:
        pipeline = wl.build_pipeline(pipeline_name)
    return pipeline

# COMMAND ----------

# Login to local Wallaroo instance

wallarooPrefix = "squishy-wallaroo-6187"
wallarooSuffix = "wallaroo.dev"
 
wl = wallaroo.Client(api_endpoint=f"https://{wallarooPrefix}.api.{wallarooSuffix}", 
                    auth_endpoint=f"https://{wallarooPrefix}.keycloak.{wallarooSuffix}", 
                    auth_type="sso")

# COMMAND ----------

workspace_name = 'housepricing'
model_name = "housepricemodel"
model_file = "./housing_model_xgb.onnx"
pipeline_name = "housing-pipe"

# COMMAND ----------

# go to housepricing workspace 
new_workspace = get_workspace(workspace_name)
_ = wl.set_current_workspace(new_workspace)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Upload the model

# COMMAND ----------

hpmodel = wl.upload_model(model_name, model_file).configure()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Upload processing modules

# COMMAND ----------

# load the preprocess module
module_pre = wl.upload_model("preprocess", "./preprocess.py").configure('python')

# load the postprocess module
module_post = wl.upload_model("postprocess", "./postprocess.py").configure('python')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Create and Deploy the Pipeline

# COMMAND ----------

pipeline = (wl.build_pipeline(pipeline_name)
              .add_model_step(module_pre)
              .add_model_step(hpmodel)
              .add_model_step(module_post)
              .deploy()
           )
pipeline

# COMMAND ----------

pipeline.undeploy()

# COMMAND ----------


