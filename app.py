# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 00:40:57 2020

@author: Swamy
"""


import numpy as np
import pickle
from flask import Flask,render_template,jsonify,request

# Initialising the flask application
app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

# Routing the application to root folder
@app.route('/')
def home():
    return render_template('index.html')

# Routing to the prediction outcome
@app.route('/predict',methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_data = [np.array(int_features)]
    prediction = model.predict_proba(final_data)
    out_var = prediction[0][1]
    Output = "{:.0%}".format(out_var)
    return render_template('index.html', prediction_text = 'Bike is Purchased {} '.format(Output))

if __name__ == '__main__':
    app.run(debug = True)
    