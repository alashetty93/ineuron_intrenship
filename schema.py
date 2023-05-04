from shipment.exception import ShipmentException
from shipment.logger import logging
from shipment import utils
import os,sys
import pandas as pd
import yaml

def create_dataset_schema():
    try:
        df = utils.get_collection_as_dataframe(
            database_name = "project",
            collection_name = "shipment"            
        )

        dataset_cols = df.dtypes.apply(lambda x:x.name).to_dict()
        #datatype_present = list(set(df_dict.values()))

        numerical_cols = {}
        categorical_cols = {}

        for key, value in  dataset_cols.items():
            if value in ['float64',"int64"]:
                numerical_cols[key] = value
            else:
                categorical_cols[key] = value
        
        schema = {}
        schema['REFERENCE DATASET COLUMNS'] = dataset_cols
        schema['REFERENCE NUMERICAL COLUMNS'] = numerical_cols
        schema['REFERENCE CATEGORICAL COLUMNS'] = categorical_cols

        schema_file_path = os.path.join(os.getcwd(),"config","schema.yaml")

        utils.write_yaml_file(file_path = schema_file_path, data = schema)

    except Exception as e:
        raise ShipmentException(e,sys)          


create_dataset_schema()
