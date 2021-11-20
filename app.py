import numpy as np
import os
from tensorflow.keras.preprocessing import image 
import pandas as pd
import cv2
import tensorflow as tf
# Flask utils
from tensorflow.keras.models import load_model
from keras.applications.vgg16 import preprocess_input
from flask import Flask,request, render_template
from werkzeug.utils import secure_filename
from tensorflow.python.keras.backend import set_session


global graph
sess = tf.compat.v1.Session()
#graph=tf.get_default_graph()
graph=tf.compat.v1.get_default_graph()

# Define a flask app
app = Flask(__name__)
set_session(sess)
# Load your trained model
model = load_model('braintumor.h5')

#print('Model loaded. Check http://127.0.0.1:5000/')


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('brainintro.html')

@app.route('/predict1', methods=['GET'])
def predict1():
    # Main page
    return render_template('brainpred2.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        df= pd.read_csv('patient.csv')
        file = request.files['image']
        filename = file.filename
        name=request.form['name']
        age=request.form['age']
        
        # Save the file to ./uploads
        
        file_path = os.path.join('uploads', filename)
        file.save(file_path)
        img = image.load_img(file_path, target_size=(64, 64))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        img_data = preprocess_input(x)
        classes=model.predict(img_data)
        result=int(classes[0][0])
        if result==0:
            text = "You are perfectly fine"
            inp = "No tumor"
        else:
            text = "You are infected! Please Consult Doctor"
            inp="Tumor detected"

        df=df.append(pd.DataFrame({'name':[name],'age':[age],'status':[inp]}),ignore_index=True)
        df.to_csv('patient.csv',index = False)
        return text

if __name__ == '__main__':
    app.run(debug=True)
