import pandas as pd
import numpy as np
import os
import sys

from src.utils import CustomException
from src.logger import logging


from src.exception import error_message_details
from src.logger import logging

from src.utils import load_file


# it will call all the model/pkl file and do preprocessing
class PredictPipeline():
    def __init__(self):
        pass
    
    def predict (self,features):
        try:
            
            model_path = os.path.join("artifacts","model.pkl")
            preprocess_path = os.path.join("artifacts","preprocessing.pkl")
            print("Before loading")
            
            model = load_file(file_path= model_path)
            preprocessor = load_file(file_path=preprocess_path)
            print("After laoding")
            
            # calling data transformation function from Data_transformation file to scale data
            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            
            return pred
        
        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData():
    def __init__(self,
                    gender=                       str,
                    race_ethnicity =              str,
                    parental_level_of_education = str ,
                    lunch =                       str,
                    test_preparation_course =     str,
                    writing_score =               int,
                    reading_score =               int
                   
                ):
        
                    self.gender  =       gender                   
                    self.race_ethnicity= race_ethnicity 
                    self.parental_level_of_education=  parental_level_of_education
                    self.lunch=  lunch                  
                    self.test_preparation_course=  test_preparation_course 
                    self.writing_score=writing_score 
                    self.reading_score= reading_score            
                    
                    
                    print(self.race_ethnicity) 
    
    def get_data_as_data_frame (self):
        
        try:
            custome_data_input_dict = {
                                        'gender': [self.gender] ,                     
                                        'race_ethnicity' : [self.race_ethnicity]  ,           
                                        'parental_level_of_education' : [self.parental_level_of_education] ,
                                        'lunch' : [self.lunch] ,                     
                                        'test_preparation_course' : [self.test_preparation_course] ,   
                                        # 'math_score' : [self.math_score] ,                
                                        'reading_score' : [self.reading_score]   ,           
                                        'writing_score' : [self.writing_score]             
                                    }
            return pd.DataFrame(custome_data_input_dict)
                    
        except Exception as e:
            raise CustomException(e,sys)
        
        

        
        

    
    