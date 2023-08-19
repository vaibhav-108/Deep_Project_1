import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd


from sklearn.model_selection import train_test_split
from dataclasses import dataclass    #--->  help to store __init__ function of class


from src.components.Data_transformation import DataTransformation
from src.components.Data_transformation import DataTransformConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import Model_Trainer

@dataclass  #--> it will hold __init__,__repr__ value of class
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts','train.csv')   # since we use dataclass decorator
    test_data_path = os.path.join('artifacts','test.csv')     # we dont need to use __init__
    raw_data_path = os.path.join('artifacts','raw.csv')
    
    
class Data_ingetion:
    
    def __init__(self) -> None:
        self.ingetion_config= DataIngestionConfig()  #--> it will call all var of that class
        
    def initiate_data_ingestion(self):
        logging.info(" Data ingetion process started")
        
        try:
            df= pd.read_csv('Notebook\data\stud.csv')
            logging.info("Read the dataset as dataframe")
            
            #create dir to save data
            os.makedirs(os.path.dirname(self.ingetion_config.train_data_path),exist_ok=True) 
            
            df.to_csv(self.ingetion_config.raw_data_path,index=False,header=True)
            
            logging.info("train_test_split initiated")
            train_set,test_set = train_test_split(df,test_size=.2,random_state=42)
            
            
            train_set.to_csv(self.ingetion_config.train_data_path,index= False,header=True)
            test_set.to_csv(self.ingetion_config.test_data_path,index= False,header=True)
            logging.info(" Train test split of data is completed")
            
            return(
                self.ingetion_config.train_data_path,
                self.ingetion_config.test_data_path
              )
            
            
        except Exception as e:
            
            raise CustomException(e,sys)
            
if __name__ == "__main__":
    obj= Data_ingetion() 
    train_data,test_data=obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transform(train_data,test_data)
    
    model_trainer= Model_Trainer()
    print(model_trainer.initiate_model_training(train_arr=train_arr,test_arr=test_arr))
    
    
    
    
    
    
    
    