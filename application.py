from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

# Ridge regression model and standar scaler pickel
ridg_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb')) 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        Temperature = float(request.form['Temperature'])
        RH = float(request.form['RH'])
        Ws = float(request.form['Ws'])
        Rain = float(request.form['Rain'])
        FFMC = float(request.form['FFMC'])
        DC = float(request.form['DMC'])
        ISI = float(request.form['ISI'])
        Classes = float(request.form['Classes'])
        Region = float(request.form['Region'])
        new_data_Scaled = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DC, ISI, Classes, Region]])
        result = ridg_model.predict(new_data_Scaled)
        return render_template('home.html', prediction_text='The predicted area of fire is: {:.2f}'.format(result[0]))


    return render_template('home.html')
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001)
