from shipment.exception import ShipmentException
from shipment.logger import logging
from shipment.predictor import ModelResolver
from datetime import datetime
from shipment import utils
import numpy as np
import pandas as pd
import os, sys

PREDICTION_DIR = "prediction"

def start_batch_prediction(input_file_path):
    try:
        os.makedirs(PREDICTION_DIR,exist_ok = True)

        logging.info(f"creating object of data transformer and model resolver")
        model_resolver = ModelResolver(model_registry = "saved_models")
        
        logging.info(f"Reading csv file:{input_file_path}")
        df = pd.read_csv(input_file_path)
        #print(df)

        logging.info(f"dropping unnecessary columns")
        new_df = drop_unwanted_columns(df = df)
        #print(new_df)

        logging.info(f"formatting text data in numerical columns")
        input_df = clean_columns_data(df = new_df)
        #print(input_df)

        logging.info(f"loading numerical imputer and input transformer to transformer dataset")
        input_transformer = utils.load_object(file_path = model_resolver.get_latest_input_transformer_path())
        # print(numerical_imputer)
        # print(input_transformer)
        
        input_feature_names = list(input_transformer.feature_names_in_)
        print(input_feature_names)
        input_arr = input_transformer.transform(input_df[input_feature_names])
        #print(input_arr)
        logging.info(f"loading model to make predictions")
        model = utils.load_object(file_path = model_resolver.get_latest_model_path())
        #print(model)
        prediction = model.predict(input_arr)

        #print(prediction.reshape(-1,1))
        logging.info(f"loading target transformer to inverse transform target variable")
        target_transformer = utils.load_object(file_path = model_resolver.get_latest_target_transformer_path())
        final_prediction = target_transformer.inverse_transform(prediction.reshape(-1,1))
        #print(final_prediction)

        df["prediction "] = prediction
        df["final_prediction"] = np.expm1(final_prediction)
        #print(df)
        prediction_file_name = os.path.basename(f"{'Prediction file_' + datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.csv")
        prediction_file_path = os.path.join(PREDICTION_DIR,prediction_file_name)
        df.to_csv(prediction_file_path,index = False,header = True)

        return prediction_file_path
    
    except Exception as e:
        ShipmentException(e,sys)

#function to drop unwanted columns
def drop_unwanted_columns(df)->pd.DataFrame:
    try:
        #drop unnecessary columns as they are not required for model training
        columns_to_drop =  ['ID', 'Project Code', 'PQ #', 'PO / SO #', 'ASN/DN #','Item Description','PQ First Sent to Client Date',
        'PO Sent to Vendor Date','Scheduled Delivery Date','Delivered to Client Date','Delivery Recorded Date']

        df = df.drop(columns=columns_to_drop)
        return df

    except Exception as e:
            raise ShipmentException(e,sys)

#clean column data
def clean_columns_data(df):
    try:
        #create a copy of dataframe
        new_df = df.copy()

        #replace text starting with "See" and "Invoiced" with np.nan in "Freight Cost" feature 
        new_df.loc[(new_df['Freight Cost (USD)'].str.contains('See')) | (new_df['Freight Cost (USD)'].str.contains('Invoiced')),'Freight Cost (USD)'] = np.nan
        
        #replace text "Freight Included in Commodity Cost" with 0 in "Freight Cost" feature
        new_df.loc[new_df['Freight Cost (USD)'].str.contains('Commodity', na = False),'Freight Cost (USD)'] = 0

        #replace text starting with "See" and "Captured" with np.nan in "Weight" feature
        new_df.loc[(new_df['Weight (Kilograms)'].str.contains('See')) | (new_df['Weight (Kilograms)'].str.contains('Captured')),'Weight (Kilograms)'] = np.nan

        #convert datatype of 'Freight Cost' and 'Weight' to float.
        new_df['Freight Cost (USD)'] = new_df['Freight Cost (USD)'].astype('float')
        new_df['Weight (Kilograms)'] = new_df['Weight (Kilograms)'].astype('float')

        return new_df
    except Exception as e:
            raise ShipmentException(e,sys)
            