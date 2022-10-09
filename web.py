# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 17:30:46 2022

@author: param
"""

from flask import Flask,render_template,request
import pandas as pd
import pickle
app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
df=pd.read_csv('MobileTest.csv')
@app.route('/',methods=['GET'])
def home():
    return render_template('home.html',data=df)
@app.route('/predict',methods=['GET','POST'])
def predict(): 
    t_ram = request.form.get('t_ram')
    t_battery_power = request.form.get('t_battery_power')
    t_px_height = request.form.get('t_px_height')
    t_px_width = request.form.get('t_px_width')
    
  
    test=pd.DataFrame([[t_ram,t_battery_power,t_px_height,t_px_width ]],columns=['ram','battery_power','px_height','px_width'])
    
    #return render_template ('result.html',prediction_text="The encoded values are post location is {}".format(test))
    
    prediction =model.predict(test)
    prediction=prediction.item()
    
    
    return render_template ('result.html',prediction_text="The price category of smartphone based on entered features is {} ".format(prediction))
    
    
if __name__=='__main__':
  app.run(port=8000)