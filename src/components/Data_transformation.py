import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import save_object




from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


#to create model and save it into pickel file
@dataclass
class DataTransformConfig:
    preprocessor_obj_file_path= os.path.join('artifacts',"preprocessing.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transform_config= DataTransformConfig()
        
    #create object which will transform your data
    def get_data_transformer_object(self):
        
    
        try:
            
            num_column= ['writing_score','reading_score' ]
            cat_column= ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            
            num_pipeline = Pipeline(
                            steps=
                            [
                                ('imputer',SimpleImputer(strategy="median")),
                                ("scalar",StandardScaler())
                    
                            ]
            )
            
            
            cat_pipeline = Pipeline(
                                steps=
                                [
                                    ('imputer',SimpleImputer(strategy="most_frequent")),
                                    ("encoding",OneHotEncoder()),
                                    ("scaling",StandardScaler(with_mean=False))
                                ]
            )
            
            logging.info("numeric column {}".format(num_column))
            logging.info(f"categorical columns {cat_column} ")
            
            
            preprocess = ColumnTransformer(
                                        [
                                          ("num_transform",num_pipeline,num_column),
                                          ("cat_transform",cat_pipeline,cat_column)
                                        ]
            )
            
            return preprocess
        
        except Exception as e:
            raise CustomException(e,sys)
        
        
    def initiate_data_transform(self,train_path,test_path):
        
        try:
            
            train_df= pd.read_csv(train_path)
            test_df= pd.read_csv(test_path)
            
            logging.info("Train and test data is readed")
            
            logging.info("Obtaining preprocess object")
            
            preprocess_obj = self.get_data_transformer_object()
            
            target_column= 'math_score'
            numerical_column = ['reading_score', 'writing_score']
            
            
            input_train_df= train_df.drop(columns=[target_column],axis=1)
            target_train_df = train_df[target_column]
            
            input_test_df= test_df.drop(columns=[target_column],axis=1)
            target_test_df = test_df[target_column]
            
            logging.info("applying preprocessing on train & test data")
            
            input_feature_train_df_arr= preprocess_obj.fit_transform(input_train_df)
            input_feature_test_df_arr = preprocess_obj.transform(input_test_df)
            
            train_arr= np.c_[input_feature_train_df_arr, np.array(target_train_df)]
            test_arr= np.c_[input_feature_test_df_arr, np.array(target_test_df)]
            
            logging.info("save preprocesssing object")
            
            
            save_object(file_path=self.data_transform_config.preprocessor_obj_file_path,
                          obj=preprocess_obj
                          )
           
            
            logging.info("pickle file is created")
            
            
            return (train_arr,test_arr,self.data_transform_config.preprocessor_obj_file_path)
                    
                    
            
            
        except Exception as e:
            raise CustomException(e,sys)
            
        
        
        
        
        
        
        
            
    

