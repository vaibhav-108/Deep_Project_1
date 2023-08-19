import os
import sys
import pickle
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score

def save_object(file_path,obj):
    
    try:
        
        dir_path= os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
    
    except Exception as e:
        raise CustomException(e,sys)     
           
    
def evaluate_model(x_train,y_train,x_test,y_test,models):
    
    try:
        
        report= {}
        model_name =[]
        for i in range(len(list(models))):
            
            model= list(models.values())[i]
            
            model.fit(x_train,y_train)
            
            y_predict_train =model.predict(x_train)
            y_predict_test =model.predict(x_test)
            
            R2_Score_train= r2_score(y_train,y_predict_train)
            R2_Score_test= r2_score(y_test,y_predict_test)
            
            report[list(models.keys())[i]] = R2_Score_test
            
            print(model_name.append(list(models.keys())[i]))
            
        return report
            
    except Exception as e:
        raise CustomException(e,sys)
            
            
            
            
            
            
            
            
            
        
    