from flask import Flask, request, render_template_string
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('PDDS.keras')

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head><title>Plant Disease Detector</title></head>
<body>
    <h1>Upload a Plant Leaf Image</h1>
    <form action="/predict" method="POST" enctype="multipart/form-data">
        <input type="file" name="file" required>
        <input type="submit" value="Predict">
    </form>
    {% if prediction %}
        <h2>{{ prediction }}</h2>
    {% endif %}
</body>
</html>
'''

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    filepath = os.path.join('/tmp', file.filename)
    file.save(filepath)

    img = preprocess_image(filepath)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)

    return render_template_string(HTML_TEMPLATE, prediction=f"Predicted Class: {predicted_class}")

