from shipment.logger import logging
from shipment.exception import ShipmentException
from shipment.entity import config_entity,artifact_entity
import os,sys
import pandas as pd
from evidently.model_profile import Profile
from evidently.metric_preset import DataDriftPreset
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab
from evidently import ColumnMapping
from evidently.model_profile import Profile
import json
import yaml

import warnings
warnings.filterwarnings("ignore")

class DataValidation:

    def __init__(self,
    data_ingestion_artifact:artifact_entity.DataIngestionArtifact,
    data_validation_config:config_entity.DataValidationConfig
    ):
        try:
            logging.info(f"{'>>'*10} Data Validation {'<<'*10}")
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            
        except Exception as e:
            raise ShipmentException(e,sys)    
    
    def get_train_and_test_df(self):
        try:
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            return train_df,test_df

        except Exception as e:
            raise ShipmentException(e,sys)

    def is_train_test_file_exists(self):
        try:
            logging.info("Checking if training and test file is available")
            is_train_file_exist = False
            is_test_file_exist = False

            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            is_train_file_exist = os.path.exists(train_file_path)
            is_test_file_exist = os.path.exists(test_file_path)

            is_available = is_train_file_exist and is_test_file_exist

            logging.info(f"Is train and test file exists?-> {is_available}")

            if not is_available:
                training_file = self.data_ingstion_artifact.train_file_path
                test_file = self.data_ingestion_artifact.test_file_path
                message=f"Training file: {training_file} or Testing file: {testing_file} is not present"
                raise Exception(message)
            
            return is_available
        
        except Exception as e:
            raise ShipmentException(e,sys)
    
    #function to drop unwanted columns
    def drop_unwanted_columns(self,df:pd.DataFrame)->pd.DataFrame:
        '''
        This function will drop the unnecessary columns in dataset

        Params:
        df: Accepts a pandas dataframe
        =================================================
        returns Pandas DataFrame
        '''
        try:
            #drop unnecessary columns as they are not required for model training
            columns_to_drop:list =  ['ID', 'Project Code', 'PQ #', 'PO / SO #', 'ASN/DN #','Item Description','PQ First Sent to Client Date',
            'PO Sent to Vendor Date','Scheduled Delivery Date','Delivered to Client Date','Delivery Recorded Date']

            df = df.drop(columns_to_drop,axis =1)
            
            return df

        except Exception as e:
                raise ShipmentException(e,sys)
              
    # def is_required_column_exist(self,base_df,current_df,report_key_name:str)-> bool:
    #     try:
    #         base_columns = base_df.columns
    #         current_columns = current_df.columns

    #         missing_columns = []

    #         for base_column in base_columns:
    #             if base_column not in current_columns:
    #                 missing_columns.append(base_column)

    #         if len(missing_columns)>0:
    #             self.validation_error[report_key_name] = missing_columns
    #             return False

    #         return True
        
        except Exception as e:
            raise ShipmentException(e,sys)

    def get_and_save_data_drift_report(self):
        try:
            logging.info(f"reading base dataframe")
            base_df = pd.read_csv(self.data_validation_config.base_file_path)

            logging.info(f"reading train and test dataframe")
            train_df,test_df = self.get_train_and_test_df()

            logging.info(f"dropping unwanted columns from  base, train and test dataset")
            base_df = self.drop_unwanted_columns(df = base_df)
            train_df = self.drop_unwanted_columns(df = train_df)
            test_df = self.drop_unwanted_columns(df = test_df)

            logging.info(f"creating data drift report for training dataset")
            train_profile = Profile(sections=[DataDriftProfileSection()])
            train_profile.calculate(base_df, train_df)
            train_report = json.loads(train_profile.json())

            logging.info(f"creating data drift report for test dataset")
            test_profile = Profile(sections=[DataDriftProfileSection()])
            test_profile.calculate(base_df, test_df)
            test_report = json.loads(test_profile.json())

            train_report_file_path = self.data_validation_config.train_report_file_path
            train_report_dir = os.path.dirname(train_report_file_path)
            os.makedirs(train_report_dir,exist_ok=True)

            with open(train_report_file_path,"w") as train_report_file:
                json.dump(train_report, train_report_file, indent=1)
            
            test_report_file_path = self.data_validation_config.test_report_file_path
            test_report_dir = os.path.dirname(test_report_file_path)
            os.makedirs(test_report_dir,exist_ok=True)

            with open(test_report_file_path,"w") as test_report_file:
                json.dump(test_report, test_report_file, indent=1)

        except Exception as e:
            raise ShipmentException(e,sys)

    def save_data_drift_report_page(self):
        try:
            base_df = pd.read_csv(self.data_validation_config.base_file_path)

            train_df,test_df = self.get_train_and_test_df()
            
            base_df = self.drop_unwanted_columns(df = base_df)
            train_df = self.drop_unwanted_columns(df = train_df)
            test_df = self.drop_unwanted_columns(df = test_df)

            train_dashboard = Dashboard(tabs=[DataDriftTab()])
            train_dashboard.calculate(base_df, train_df)

            train_report_page_file_path = self.data_validation_config.train_report_page_file_path
            train_report_page_dir = os.path.dirname(train_report_page_file_path)
            os.makedirs(train_report_page_dir,exist_ok=True)

            train_dashboard.save(train_report_page_file_path)

            test_dashboard = Dashboard(tabs=[DataDriftTab()])
            test_dashboard.calculate(base_df, train_df)

            test_report_page_file_path = self.data_validation_config.test_report_page_file_path
            test_report_page_dir = os.path.dirname(test_report_page_file_path)
            os.makedirs(test_report_page_dir,exist_ok=True)

            test_dashboard.save(test_report_page_file_path)

        except Exception as e:
            raise ShipmentException(e,sys)

    def initiate_data_validation(self):
        try:
            self.is_train_test_file_exists()
            self.get_and_save_data_drift_report()
            self.save_data_drift_report_page()

            data_validation_artifact = artifact_entity.DataValidationArtifact(
                train_report_file_path=self.data_validation_config.train_report_file_path,
                test_report_file_path=self.data_validation_config.test_report_file_path,
                train_report_page_file_path=self.data_validation_config.train_report_page_file_path,
                test_report_page_file_path=self.data_validation_config.test_report_page_file_path,
            )
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise ShipmentException(e,sys)