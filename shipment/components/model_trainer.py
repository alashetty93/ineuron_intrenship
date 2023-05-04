from shipment.entity import config_entity,artifact_entity,model_finder
from shipment.exception import ShipmentException
from shipment.logger import logging
import os,sys
import pandas as pd
import numpy as np
from shipment import utils
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV

class ModelTrainer:

    def __init__(self,
    model_trainer_config:config_entity.ModelTrainerConfig,
    data_transformation_artifact:artifact_entity.DataTransformationArtifact):
        
        try:
            logging.info(f"{'>>'*10} Model Trainer {'<<'*10}")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact

        except Exception as e:
            raise ShipmentException(e, sys)
    
    def train_model(self,X,y):
        try:
            model_rf = RandomForestRegressor(
                n_estimators = 80,
                min_samples_leaf = 25,
                max_depth = 25,
                criterion = 'friedman_mse',
                random_state = 0
            )

            voting_reg = VotingRegressor([
                ('knn',KNeighborsRegressor(n_neighbors = 10)),
                ('rf',model_rf)
            ])

            voting_reg.fit(X,y)

            return voting_reg

        except Exception as e:
            raise ShipmentException(e, sys)
    
    def initiate_model_trainer(self)->artifact_entity.ModelTrainerArtifact:
        try:
            logging.info(f"loading train and test array")
            train_arr = utils.load_numpy_array_data(file_path = self.data_transformation_artifact.tranformed_train_path)
            test_arr = utils.load_numpy_array_data(file_path = self.data_transformation_artifact.tranformed_test_path)
            
            logging.info(f"split input and target feature from train and test array")
            X_train, y_train = train_arr[:,:-1], train_arr[:,-1]
            X_test, y_test = test_arr[:,:-1], test_arr[:,-1]
            
            logging.info(f"train the model")
            model = self.train_model(X = X_train,y = y_train)

            logging.info(f"calculate train r2 score") 
            yhat_train = model.predict(X_train)
            train_r2_score = r2_score(y_true = y_train, y_pred = yhat_train)

            logging.info(f"calculate test r2 score") 
            yhat_test = model.predict(X_test)
            test_r2_score = r2_score(y_true = y_test, y_pred = yhat_test)

            logging.info(f"train score: {train_r2_score} and test score: {test_r2_score}")
            #check for expected score
            logging.info(f"check if model is underfitted or not")
            if test_r2_score < self.model_trainer_config.expected_score:
                raise Exception(f"Model is not good as it is not able to give\
                    expected accuracy: {self.model_trainer_config.expected_score}: model actual score: {test_r2_score}")

            #check for overfitting or underfitting
            logging.info(f"check if model is overfitted or not")
            diff = abs(train_r2_score - test_r2_score)
            if diff > self.model_trainer_config.overfitting_threshold:
                raise Exception(f"Train and test score difference is more than overfitting threshold: {self.model_trainer_config.overfitting_threshold * 100}%")

            #save the training model
            logging.info(f"save model object")
            utils.save_object(file_path = self.model_trainer_config.model_path, object = model)

            #prepare artifact
            logging.info(f"prepare the artifact")
            model_trainer_artifact = artifact_entity.ModelTrainerArtifact(
                model_path = self.model_trainer_config.model_path,
                train_r2_score = train_r2_score,
                test_r2_score = test_r2_score
            )
            
            return model_trainer_artifact
        
        except Exception as e:
            raise ShipmentException(e, sys)