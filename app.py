from flask import Flask,request,render_template
import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler

from src.pipeline.predict_pipeline import CustomData,PredictPipeline



application= Flask(__name__)
app= application

@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    
    if request.method == 'GET':
        return render_template('home.html')
    
    else:
        #take input from fronend and call function from predict pipeline
        data = CustomData(
                        gender=                       request.form.get('gender'),
                        race_ethnicity =              request.form.get('ethnicity'),
                        parental_level_of_education = request.form.get('parental_level_of_education'),
                        lunch =                       request.form.get('lunch'),
                        test_preparation_course =     request.form.get('test_preparation_course'),
                        # math_score =                  request.form.get('math_score'),
                        reading_score =               request.form.get('reading_score'),
                        writing_score =               request.form.get('writing_score')

                )
        
        
        pred_df= data.get_data_as_data_frame()
        print(pred_df)
        print("Before predict data")
        
        predict_pipeline = PredictPipeline()
        
        result= predict_pipeline.predict(pred_df)
        
        return render_template('home.html', results= result[0])
    
    
if __name__=="__main__":
    app.run(host="0.0.0.0") 
