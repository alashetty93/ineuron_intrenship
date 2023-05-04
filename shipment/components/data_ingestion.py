import os,sys
from shipment.exception import ShipmentException
from shipment.logger import logging
from shipment.entity import config_entity,artifact_entity
from shipment.utils import get_collection_as_dataframe
from sklearn.model_selection import train_test_split

class DataIngestion:

    def __init__(self,data_ingestion_config:config_entity.DataIngestionConfig):
        
        try:
            logging.info(f"{'>>'*10} Data Ingestion {'<<'*10}")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise ShipmentException(e,sys)
    
    def initiate_data_ingestion(self):
        try:        
            df:pd.DataFrame = get_collection_as_dataframe(
                database_name = self.data_ingestion_config.database_name,
                collection_name = self.data_ingestion_config.collection_name
            )

            feature_store_dir = os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_dir,exist_ok = True)

            df.to_csv(path_or_buf = self.data_ingestion_config.feature_store_file_path,index = False,header = True)

            train_df,test_df = train_test_split(df,test_size = self.data_ingestion_config.test_size,random_state = 42)

            dataset_dir = os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dataset_dir,exist_ok = True)

            train_df.to_csv(path_or_buf = self.data_ingestion_config.train_file_path,index = False,header = True)
            test_df.to_csv(path_or_buf = self.data_ingestion_config.test_file_path,index = False,header = True)

            data_ingstion_artifact = artifact_entity.DataIngestionArtifact(
                feature_store_file_path = self.data_ingestion_config.feature_store_file_path,
                train_file_path = self.data_ingestion_config.train_file_path,
                test_file_path = self.data_ingestion_config.test_file_path
            )

            return data_ingstion_artifact
        
        except Exception as e:
            raise ShipmentException(e,sys)



