import os
import sys
from dataclasses import dataclass


from sklearn.ensemble import (RandomForestRegressor,
                            GradientBoostingRegressor,
                            AdaBoostRegressor,
                                
)

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_model



@dataclass
class ModelTrainerConfig:
    model_TrainerConfig_path= os.path.join("artifacts","model.pkl")
    
class Model_Trainer:
    def __init__(self):
        self.model_trainer_config= ModelTrainerConfig()
        
    def initiate_model_training(self,train_arr,test_arr):
        
        try:
            
            logging.info(" train & test split from train & test_arr which is return/output from Data_transform")
            
            #since return of Data_transform train_arr & test_arr contain target column we need to split it
            X_train,y_train,X_test,y_test= (
                                            train_arr[:,:-1],
                                            train_arr[:,-1],
                                            test_arr[:,:-1],
                                            test_arr[:,-1]
                                            )
            
            
            models= {
                    
                    'K-NeighboursN'         : KNeighborsRegressor(),
                    'Decision_tree'         : DecisionTreeRegressor(),
                    'Random_Forest'         : RandomForestRegressor(),
                    'GradientBoost'         : GradientBoostingRegressor(),
                    'linear_regression'     : LinearRegression(),
                    'XGBRegressor'          : XGBRegressor(),
                    'CatBoost_regression'   : CatBoostRegressor(),
                    'AdaBoost'              : AdaBoostRegressor()       
            }
            
            
            
            #for hyperparameter tuning
            logging.info("Hyper parameter tuning start")
            params={
                
                "K-NeighboursN" :{},
                "Decision_tree": {
                        'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                        # 'splitter':['best','random'],
                        # 'max_features':['sqrt','log2'],
                    },
                "Random_Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "GradientBoost":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "linear_regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoost_regression":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
                
            }

            #create dict where all evaluate function values will be stored
            model_report:dict= evaluate_model(x_train=X_train,y_train=y_train,x_test=X_test,y_test=y_test,models=models,param=params)
            
            logging.info(f'model_report:- {model_report}')
            
            # to get best model score form model report
            best_score = max(sorted(model_report.values()))
            
            # to get best model name from model report
            best_model_name= list(model_report.keys())[list(model_report.values()).index(best_score)]
            
            best_model= models[best_model_name]
            logging.info(f'best_model:- {best_model}')
            
            if best_score < 0.6:
                # raise CustomException("No bets model found")
                logging.info(" no best model found")
                
            
            logging.info(" Best found model on both tarininf & testing data")
            
            
            save_object(file_path=self.model_trainer_config.model_TrainerConfig_path,
                        obj=best_model)
            
            
            predicted = best_model.predict(X_test)
            
            R2_score= r2_score(y_test,predicted)
            
            logging.info(" Return R2 score of best model")
            
            return R2_score

        except Exception as e:
            raise CustomException(e,sys)
            
        
        

