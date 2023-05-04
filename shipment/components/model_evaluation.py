from shipment.entity import config_entity,artifact_entity
from shipment.exception import ShipmentException
from shipment.logger import logging
from shipment.predictor import ModelResolver
from shipment import utils
from shipment.components.data_transformation import DataTransformation,NumericalImputationMICE
from sklearn.metrics import r2_score
import os,sys
import pandas as pd
from shipment.config import TARGET_COLUMN

class ModelEvaluation:

    def __init__(self,
    data_ingestion_artifact:artifact_entity.DataIngestionArtifact,
    data_transformation_artifact:artifact_entity.DataTransformationArtifact,
    model_trainer_artifact:artifact_entity.ModelTrainerArtifact,
    model_eval_config:config_entity.ModelEvaluationConfig   
    ):

        try:
            logging.info(f"{'>>'*10} Model Evaluation {'<<'*10}")
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_eval_config = model_eval_config
            self.model_resolver = ModelResolver()
            self.data_transformer = DataTransformation(
                data_ingestion_artifact = data_ingestion_artifact,
                data_transformation_config = config_entity.DataTransformationConfig
            )

        except Exception as e:
            ShipmentException(e,sys)

    def initiate_model_evaluation(self)->artifact_entity.ModelEvaluationArtifact:

        try:
            logging.info("if saved model folder has model the we will compare")
            latest_dir_path = self.model_resolver.get_latest_dir_path()

            if latest_dir_path is None:
                model_eval_artifact = artifact_entity.ModelEvaluationArtifact(
                    is_model_accepted = True, improved_score = None
                )
                logging.info(f"Model Evaluation Artifact:{model_eval_artifact}")
                return model_eval_artifact

            #Finding location of input transformer, target transformer and model
            logging.info(f"finding location of input transformer, target transformer and model")
            model_path = self.model_resolver.get_latest_model_path()
            input_transformer_object_path = self.model_resolver.get_latest_input_transformer_path()
            target_transformer_object_path = self.model_resolver.get_latest_target_transformer_path()

            logging.info(f"Previous trained objects of transformer, model and target encoder")
            #load previously trained objects
            model = utils.load_object(file_path = model_path)
            input_transformer = utils.load_object(file_path = input_transformer_object_path)
            target_transformer = utils.load_object(file_path = target_transformer_object_path)

            logging.info(f"Currently trained model objects")
            #currently trained model objects
            current_model = utils.load_object(file_path = self.model_trainer_artifact.model_path)
            current_input_transformer = utils.load_object(file_path = self.data_transformation_artifact.input_transformer_object_path)
            current_target_transformer = utils.load_object(file_path = self.data_transformation_artifact.target_transformer_object_path)

            #get test dataset
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            dropped_columns_test_df = self.data_transformer.drop_unwanted_columns(df = test_df)
            clean_columns_test_df = self.data_transformer.clean_columns_data(df = dropped_columns_test_df)

            #logging.info(f"Filling na with 9007 in target")
            #r2_score using previously trained model
            target_df = clean_columns_test_df[TARGET_COLUMN].fillna(9007)
            #logging.info(f"target df: {target_df}")
            y_true = target_transformer.transform(target_df.values.reshape(-1,1))
            #print(y_true)

            logging.info(f"converting input features to array")
            input_feature_name = list(input_transformer.feature_names_in_)
            #logging.info(f"input feature name: {input_feature_name}")
            #logging.info(f"test_df: {clean_columns_test_df[input_feature_name]}")
            input_arr = input_transformer.transform(clean_columns_test_df[input_feature_name])
            
            #logging.info(f"input_arr:{input_arr}")
            y_pred  = model.predict(input_arr)


            logging.info("making prediction using previous model")
            previous_model_score = round(r2_score(y_true = y_true, y_pred = y_pred),4)
            logging.info(f"Score using previous model:{previous_model_score}")

            #r2_score using current trained model
            target_df = clean_columns_test_df[TARGET_COLUMN].fillna(9007)
            y_true = target_transformer.transform(target_df.values.reshape(-1,1))

            input_feature_name = list(current_input_transformer.feature_names_in_)
            input_arr = current_input_transformer.transform(clean_columns_test_df[input_feature_name])
            current_y_pred  = current_model.predict(input_arr)

            current_model_score = round(r2_score(y_true = y_true, y_pred = current_y_pred),4)
            logging.info(f"Score using current model:{current_model_score}")

            if current_model_score<=previous_model_score:
                logging.info(f"Current trained model is not better than previous model")
                raise Exception("Current trained model is not better than previous model")

            model_eval_artifact = artifact_entity.ModelEvaluationArtifact(
                is_model_accepted = True,
                improved_score = current_model_score - previous_model_score
            )            

            logging.info(f"Model eval artifact: {model_eval_artifact}")
            return model_eval_artifact

        except Exception as e:
            raise ShipmentException(e,sys)