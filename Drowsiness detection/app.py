
from flask import Flask, render_template, request
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    model = load_model("drowiness.h5")
    labels = ['Closed', 'no_yawn', 'Open', 'yawn']
    
    if 'file' not in request.files:
        return "No file part"
    
    uploaded_file = request.files['file']
    
    if uploaded_file.filename == '':
        return "No selected file"
    
    image1 = Image.open(uploaded_file)
    image1 = image1.resize((80, 80))
    img_array = image.img_to_array(image1)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    predictions = model.predict(img_array)
    predicted_label_index = np.argmax(predictions)
    
    if predicted_label_index == 0:
        label = 'Drowsiness Detected'
    elif predicted_label_index == 3:
        label = 'Drowsiness Detected'
    elif predicted_label_index == 2 and predicted_label_index == 3:
        label = 'Drowsiness Detected'
    elif predicted_label_index == 0 and predicted_label_index == 1:
        label = 'Drowsiness Detected'
    elif predicted_label_index == 1 and predicted_label_index == 2:
        label = 'No Drowsiness Detected'
    elif predicted_label_index == 0 and predicted_label_index == 3:
        label = 'Drowsiness Detected'
    elif predicted_label_index == 1:
        label = 'No Drowsiness Detected'
    else:
        label = "No Drowsiness Detected"
    
    return render_template('result.html', label=label)

if __name__ == '__main__':
    app.run(debug=True)
