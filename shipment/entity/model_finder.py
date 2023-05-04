from shipment.entity import config_entity,artifact_entity
from shipment.exception import ShipmentException
from shipment.logger import logging
import os,sys
import pandas as pd
import numpy as np
from shipment import utils
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

class SelectBestModel:
    """
    This class shall  be used to find the model with best accuracy and AUC score.
    Written By: iNeuron Intelligence
    Version: 1.0
    Revisions: None
    """
    def __init__(self):
        pass

    def get_best_params_for_knn(self,X,y):
        try:
            #initializing with different combination of parameters
            self.param_grid = {
                'n_neighbors':[1,3,5,10,15,20]
            }

            #creating an object of the Grid Search class
            self.grid= GridSearchCV(estimator = KNeighborsRegressor(), param_grid = self.param_grid, cv = 5, n_jobs = -1)
            
            #finding the best parameters
            self.grid.fit(X, y)

            # extracting the best parameters
            self.n_neighbors = self.grid.best_params_['n_neighbors']

            #creating a new model with the best parameters
            self.knn = KNeighborsRegressor(n_neighbors = self.n_neighbors)

            self.knn.fit(X,y)

            return self.knn 

        except Exception as e:
            raise ShipmentException(e,sys)
    
    def get_best_params_for_DecisionTree(self,X,y):
        try:
            #initializing with different combination of parameters
            self.param_grid = {
                'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                'splitter':['best','random'],
                'ccp_alpha': [0.03,0.05,0.06,0.08,0.09]
            }

            #creating an object of the Grid Search class
            self.grid= GridSearchCV(estimator = DecisionTreeRegressor(), param_grid = self.param_grid, cv = 5, n_jobs = -1)
            
            #finding the best parameters
            self.grid.fit(X, y)

            # extracting the best parameters
            self.criterion = self.grid.best_params_['criterion']
            self.splitter = self.grid.best_params_['splitter']
            self.ccp_alpha = self.grid.best_params_['ccp_alpha']

            #creating a new model with the best parameters
            self.decision_tree = DecisionTreeRegressor(criterion = self.criterion, splitter = self.splitter, ccp_alpha = self.ccp_alpha)
            
            self.decision_tree.fit(X,y)

            return self.decision_tree 

        except Exception as e:
            raise ShipmentException(e,sys)
    
    def get_best_params_for_RandomForest(self,X,y):
        try:
            #initializing with different combination of parameters
            self.param_grid = {
                'n_estimators':[50,80,100,120],
                'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
                'ccp_alpha': [0.03,0.05,0.06,0.08,0.09]
            }

            #creating an object of the Grid Search class
            self.grid= GridSearchCV(estimator = RandomForestRegressor(), param_grid = self.param_grid, cv = 5, n_jobs = -1)
            
            #finding the best parameters
            self.grid.fit(X, y)

            # extracting the best parameters
            self.n_estimators = self.grid.best_params_['n_estimators']
            self.criterion = self.grid.best_params_['criterion']
            self.ccp_alpha = self.grid.best_params_['ccp_alpha']

            #creating a new model with the best parameters
            self.random_forest = RandomForestRegressor(n_estimators = self.n_estimators, criterion = self.criterion, ccp_alpha = self.ccp_alpha)

            self.random_forest.fit(X,y)

            return self.random_forest

        except Exception as e:
            raise ShipmentException(e,sys)

    def get_best_params_for_AdaBoost(self,X,y):
        try:
            #initializing with different combination of parameters
            self.param_grid = {
                'n_estimators':[50,80,100,120],
                'learning_rate':[0.01,0.05,0.1,2,5,10]
            }

            #creating an object of the Grid Search class
            self.grid= GridSearchCV(estimator = AdaBoostRegressor(), param_grid = self.param_grid, cv = 5, n_jobs = -1)
            
            #finding the best parameters
            self.grid.fit(X, y)

            # extracting the best parameters
            self.n_estimators = self.grid.best_params_['n_estimators']
            self.learning_rate = self.grid.best_params_['learning_rate']

            #creating a new model with the best parameters
            self.ada_boost = AdaBoostRegressor(n_estimators = self.n_estimators, learning_rate = self.learning_rate)

            self.ada_boost.fit(X,y)

            return self.ada_boost

        except Exception as e:
            raise ShipmentException(e,sys)
    
    def get_best_params_for_XGBoost(self,X,y):
        try:
            #initializing with different combination of parameters
            self.param_grid = {
                'n_estimators':[50,80,100,120],
                'learning_rate':[0.01,0.05,0.1,2,5,10],
                'gamma':[0.5,0.6,0.7,0.8]
            }

            #creating an object of the Grid Search class
            self.grid= GridSearchCV(estimator = XGBRegressor(), param_grid = self.param_grid, cv = 5, n_jobs = -1)
            
            #finding the best parameters
            self.grid.fit(X, y)

            # extracting the best parameters
            self.n_estimators = self.grid.best_params_['n_estimators']
            self.learning_rate = self.grid.best_params_['learning_rate']
            self.gamma = self.grid.best_params_['gamma']

            #creating a new model with the best parameters
            self.xgboost = XGBRegressor(n_estimators = self.n_estimators, learning_rate = self.learning_rate, gamma = self.gamma)

            self.xgboost.fit(X,y)
            
            return self.xgboost

        except Exception as e:
            raise ShipmentException(e,sys)
    
    def get_best_model(self,X_train,y_train, X_test, y_test):
        try:
            train_score_dict = {}
            test_score_dict = {}

            #fit knn regressor
            self.knn = self.get_best_params_for_knn(X_train,y_train)
            
            #predict result on train data
            train_pred_knn = self.knn.predict(X_train)
            #calculate r2_score with knn
            train_r2_score_knn = r2_score(y_true = y_train,y_pred = train_pred_knn)
            train_score_dict['K Neighbors Regressor'] = train_r2_score_knn

            #predict result on test data
            test_pred_knn = self.knn.predict(X_test)
            #calculate r2_score with knn
            test_r2_score_knn = r2_score(y_true = y_test,y_pred = test_pred_knn)
            test_score_dict['K Neighbors Regressor'] = test_r2_score_knn

            #fit Decision Tree Regressor
            self.decision_tree = self.get_best_params_for_DecisionTree(X_train,y_train)

            #predict result on train data
            train_pred_dec_tree = self.decision_tree.predict(X_train)
            #calculate r2_score with decision tree
            train_r2_score_dec_tree = r2_score(y_true = y_train,y_pred = train_pred_dec_tree)
            train_score_dict['Decision Tree Regressor'] = train_r2_score_dec_tree

            #predict result on test data
            test_pred_dec_tree = self.decision_tree.predict(X_test)
            #calculate r2_score with decision tree
            test_r2_score_dec_tree = r2_score(y_true = y_test,y_pred = test_pred_dec_tree)
            test_score_dict['Decision Tree Regressor'] = test_r2_score_dec_tree

            #fit Random Forest Regressor
            self.random_forest = self.get_best_params_for_RandomForest(X_train,y_train)

            #predict result on train data
            train_pred_random_forest = self.random_forest.predict(X_test)
            #calculate r2_score with random forest
            train_r2_score_random_forest = r2_score(y_true = y_train,y_pred = train_pred_random_forest)
            train_score_dict['Random Forest Regressor'] = train_r2_score_random_forest

            #predict result on test data
            test_pred_random_forest = self.random_forest.predict(X_test)
            #calculate r2_score with random forest
            test_r2_score_random_forest = r2_score(y_true = y_test,y_pred = test_pred_random_forest)
            test_score_dict['Random Forest Regressor'] = test_r2_score_random_forest

            #fit Ada Boost Regressor
            self.ada_boost = self.get_best_params_for_AdaBoost(X_train,y_train)

            #predict result on train data
            train_pred_ada_boost = self.ada_boost.predict(X_train)
            #calculate r2_score with ada boost
            train_r2_score_ada_boost = r2_score(y_true = y_train,y_pred = train_pred_ada_boost)
            train_score_dict['Ada Boost Regressor'] = train_r2_score_ada_boost

            #predict result on test data
            test_pred_ada_boost = self.ada_boost.predict(X_test)
            #calculate r2_score with ada boost
            test_r2_score_ada_boost = r2_score(y_true = y_test,y_pred = test_pred_ada_boost)
            test_score_dict['Ada Boost Regressor'] = test_r2_score_ada_boost

            #fit XGBoost Regressor
            self.xgboost = self.get_best_params_for_XGBoost(X,y)

            #predict result on train data
            train_pred_xgboost = self.xgboost.predict(X_train)
            #calculate r2_score with xgboost
            train_r2_score_xgboost = r2_score(y_true = y_train,y_pred = train_pred_xgboost)
            train_score_dict['XGB Regressor'] = train_r2_score_xgboost

            #predict result on test data
            test_pred_xgboost = self.xgboost.predict(X_test)
            #calculate r2_score with xgboost
            test_r2_score_xgboost = r2_score(y_true = y_test,y_pred = test_pred_xgboost)
            test_score_dict['XGB Regressor'] = test_r2_score_xgboost

            #get best model name based on r2 score
            best_model = max(test_score_dict,score_dict.get) 

            return best_model,train_score_dict[best_model],test_score_dict[best_model]
            
        except Exception as e:
            raise ShipmentException(e,sys)