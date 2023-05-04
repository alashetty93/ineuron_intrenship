import os, sys
from shipment.entity import config_entity, artifact_entity
from shipment.exception import ShipmentException
from typing import Optional

class ModelResolver:

    def __init__(self,
    model_registry:str = "saved_models",
    numerical_imputer_dir_name:str = "numerical_imputer", 
    input_transformer_dir_name:str = "input_transformer",
    target_transformer_dir_name:str = "target_transformer",
    model_dir_name:str = "model"
    ):

        self.model_registry = model_registry
        os.makedirs(self.model_registry,exist_ok = True)
        self.numerical_imputer_dir_name = numerical_imputer_dir_name
        self.input_transformer_dir_name = input_transformer_dir_name
        self.target_transformer_dir_name = target_transformer_dir_name
        self.model_dir_name = model_dir_name
    
    def get_latest_dir_path(self,)->Optional[str]:
        '''
        This function will give path of earlier saved model
        '''
        try:
            dir_names = os.listdir(self.model_registry)
            if len(dir_names) == 0:
                return None
            
            dir_names = list(map(int,dir_names))
            latest_dir_name = max(dir_names)

            return os.path.join(self.model_registry,f"{latest_dir_name}")

        except Exception as e:
            raise ShipmentException(e,sys)

    def get_latest_model_path(self,):
        try:
            latest_dir = self.get_latest_dir_path()

            if latest_dir is None:
                return Exception(f"Model is not available")

            model_file_path = os.path.join(latest_dir,self.model_dir_name,config_entity.model_file_name)
            return model_file_path

        except Exception as e:
            raise ShipmentException(e,sys)
    
    def get_latest_numerical_imputer_path(self,):
        try:
            latest_dir = self.get_latest_dir_path()

            if latest_dir is None:
                return Exception(f"Numerical Imputer is not available")

            numerical_imputer_file_path = os.path.join(latest_dir,self.numerical_imputer_dir_name,config_entity.numerial_imputer_object_file_name)
            return numerical_imputer_file_path

        except Exception as e:
            raise ShipmentException(e,sys)

    def get_latest_input_transformer_path(self,):
        try:
            latest_dir = self.get_latest_dir_path()

            if latest_dir is None:
                return Exception(f"Transformer is not available")

            input_transformer_file_path = os.path.join(latest_dir,self.input_transformer_dir_name,config_entity.input_transformer_object_file_name)
            return input_transformer_file_path 

        except Exception as e:
            raise ShipmentException(e,sys)    
    
    def get_latest_target_transformer_path(self,):
        try:
            latest_dir = self.get_latest_dir_path()

            if latest_dir is None:
                return Exception(f"Transformer is not available")

            target_transformer_file_path = os.path.join(latest_dir,self.target_transformer_dir_name,config_entity.target_transformer_object_file_name)
            return target_transformer_file_path 

        except Exception as e:
            raise ShipmentException(e,sys)  
    
    def get_latest_save_dir_path(self):
        '''
        This function will create path for latest model to save
        '''
        try:
            latest_dir = self.get_latest_dir_path()
            
            if latest_dir == None:
                return os.path.join(self.model_registry,f"{0}")

            latest_dir_name = int(os.path.basename(self.get_latest_dir_path()))

            new_model_file_path = os.path.join(self.model_registry,f"{latest_dir_name+1}")
            return new_model_file_path
        
        except Exception as e:
            raise ShipmentException(e,sys)

    def get_latest_save_model_path(self):
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir, self.model_dir_name,config_entity.model_file_name)

        except Exception as e:
            raise ShipmentException(e, sys)

    def get_latest_save_numerical_imputer_path(self):
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir, self.numerical_imputer_dir_name,config_entity.numerial_imputer_object_file_name)

        except Exception as e:
            raise ShipmentException(e, sys)
    
    def get_latest_save_input_transfomer_path(self):
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir, self.input_transformer_dir_name,config_entity.input_transformer_object_file_name)

        except Exception as e:
            raise ShipmentException(e, sys)
    
    def get_latest_save_target_transfomer_path(self):
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir, self.target_transformer_dir_name,config_entity.target_transformer_object_file_name)

        except Exception as e:
            raise ShipmentException(e, sys)