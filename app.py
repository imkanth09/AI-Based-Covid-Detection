from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer
#from flask_ngrok import run_with_ngrok
import pickle

# Define a flask app
app = Flask(__name__)
#run_with_ngrok(app)

# Model saved with Keras model.save()
#MODEL_PATH = 'C:\Users\Krishnakanth\Downloads\AI Based Smart Covid Predictor\model'

#Load your trained model
model = load_model('model/Covid_Xray.h5')
#model = pickle.load(open('model/model.pkl', 'rb'))
#model._make_predict_function()          # Necessary to make everything ready to run on the GPU ahead of time
print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(150,150)) #target_size must agree with what the trained model expects!!

    #Preprocessing the image
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

   
    pred = model.predict(img)
    prob= model.predict_proba(img)[0][0]
    return prob


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        dir='static/upload'
        basepath = os.path.dirname(dir)
        file_path = os.path.join(
           basepath, 'upload', secure_filename(f.filename))
        f.save(file_path)
        preds = model_predict(file_path, model)
        if preds>0.5:
            prediction="NORMAL"
        else:
            prediction="COVID"

     
        #ind = training_set.class_indices

        
        # Save the file to ./uploads
         #basepath = os.path.dirname(__file__)
        #file_path = os.path.join(
           # basepath, 'uploads', secure_filename(f.filename))
        #f.save(file_path)

        # Make prediction
        #preds = model_predict(file_path, model)
        #os.remove(file_path)#removes file from the server after prediction has been returned
        

        # Arrange the correct return according to the model. 
        # In this model 1 is Pneumonia and 0 is Normal.
        #str1 = 'Pneumonia'
        #str2 = 'Normal'
        #if preds == 1:
           # return str1
        #else:
            #return str2
    return prediction

    #this section is used by gunicorn to serve the app on Heroku
if __name__ == '__main__':
        app.run()
    #uncomment this section to serve the app locally with gevent at:  http://localhost:5000
    # Serve the app with gevent 