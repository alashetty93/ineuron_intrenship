from shipment.config import mongo_client
from shipment.logger import logging
from shipment.exception import ShipmentException
import pandas as pd
import os,sys
import numpy as np
import dill
import yaml

def get_collection_as_dataframe(database_name:str,collection_name:str)-> pd.DataFrame:
    '''
    Description: This function coonvert database collection into dataframe.

    Params:
    database_name:database name
    collection_name: collecttion name
    ======================================
    returns: pandas dataframe of a collection.
    '''

    try:
        logging.info(f"fetching data from database: {database_name} and collection: {collection_name}")
        #fetch the records from mongodb collection 
        mongo_record = list(mongo_client[database_name][collection_name].find())

        logging.info(f"creating dataframe")
        #create dataframe
        df = pd.DataFrame(mongo_record)

        logging.info(f"found columns: {df.columns}")
        #drop _id column that is provided by mongodb by default
        if "_id" in df:
            
            logging.info(f"dropping column: _id")
            df.drop(columns = ['_id'],inplace = True)

        return df
    
    except Exception as e:
        raise ShipmentException(e,sys) 

def write_yaml_file(file_path:str,data:dict):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok = True)

        with open(file_path,'w') as yaml_file:
            if data is not None:
                yaml.dump(data,yaml_file,sort_keys=False, default_flow_style=False)

    except Exception as e:
        raise ShipmentException(e,sys)

def read_yaml_file(file_path:str)->dict:
    """
    Reads a YAML file and returns the contents as a dictionary.
    file_path: str
    """
    try:
        with open(file_path,'rb') as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise ShipmentException(e,sys)

def save_numpy_array_data(file_path:str, array:np.ndarray):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok = True)

        with open(file_path,'wb') as file_object:
            np.save(file_object,array)
    
    except Exception as e:
        raise ShipmentException(e,sys)

def load_numpy_array_data(file_path:str)->np.ndarray:
    """
    load numpy array data file
    file_path: str path where file is located.
    array: np.array data loaded
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok = True)

        with open(file_path,'rb') as file_object:
            return np.load(file_object)
    
    except Exception as e:
        raise ShipmentException(e,sys)

def save_object(file_path:str,object):
    """
    file_path: str
    obj: Any sort of object
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(object, file_obj)

    except Exception as e:
        raise ShipmentException(e,sys)

def load_object(file_path:str):
    """
    file_path: str
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise ShipmentException(e,sys)