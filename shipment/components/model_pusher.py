from shipment.predictor import ModelResolver
from shipment.entity import config_entity, artifact_entity
from shipment.logger import logging
from shipment.exception import ShipmentException
import os, sys
from shipment import utils

class ModelPusher:

    def __init__(self,
    model_pusher_config:config_entity.ModelPusherConfig,
    data_transformation_artifact:artifact_entity.DataTransformationArtifact,
    model_trainer_artifact:artifact_entity.ModelTrainerArtifact):

        try:
            logging.info(f"{'>>'*10} Model Pusher {'<<'*10}")
            self.model_pusher_config = model_pusher_config
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_resolver = ModelResolver(model_registry=self.model_pusher_config.outside_saved_model_dir)

        except Exception as e:
                raise ShipmentException(e, sys)
    
    def initiate_model_pusher(self,)->artifact_entity.ModelPusherArtifact:
        try:
            #load 
            logging.info(f"loading numerical imputer, input transformer, target transformer and model")
            input_transformer_object = utils.load_object(file_path = self.data_transformation_artifact.input_transformer_object_path)
            target_transformer_object = utils.load_object(file_path = self.data_transformation_artifact.target_transformer_object_path)
            model = utils.load_object(file_path = self.model_trainer_artifact.model_path)

            #model pusher dir
            logging.info(f"Saving model into model pusher directory")
            utils.save_object(file_path = self.model_pusher_config.pusher_input_transformer_path, object = input_transformer_object)
            utils.save_object(file_path = self.model_pusher_config.pusher_target_transformer_path, object = target_transformer_object)
            utils.save_object(file_path = self.model_pusher_config.pusher_model_path, object = model)

            #saved model dir
            logging.info(f"Saving model in saved model dir")
            input_transfomer_path = self.model_resolver.get_latest_save_input_transfomer_path()
            target_transfomer_path = self.model_resolver.get_latest_save_target_transfomer_path()
            model_path = self.model_resolver.get_latest_save_model_path()

            utils.save_object(file_path = input_transfomer_path, object = input_transformer_object)
            utils.save_object(file_path = target_transfomer_path, object = target_transformer_object)
            utils.save_object(file_path = model_path, object = model)

            model_pusher_artifact = artifact_entity.ModelPusherArtifact(
                pusher_model_dir = self.model_pusher_config.pusher_model_dir,
                outside_saved_model_dir = self.model_pusher_config.outside_saved_model_dir
            )

            logging.info(f"Model pusher artifact: {model_pusher_artifact}")
            return model_pusher_artifact

        except Exception as e:
            raise ShipmentException(e,sys)  