from flask import Flask, request
import numpy as np
import tensorflow as tf
import requests
from PIL import Image
import io
import os

# Load MobileNetV2
inception_net = tf.keras.applications.MobileNetV2()

# Load human-readable labels for ImageNet from disk or internet
try:
    with open('imagenet_labels.txt', 'r') as f:
        labels = f.read().split('\n')
except FileNotFoundError:
    response = requests.get("https://git.io/JJkYN")
    labels = response.text.split("\n")

def classify_image(inp):
    inp = inp.reshape((-1, 224, 224, 3))
    inp = tf.keras.applications.mobilenet_v2.preprocess_input(inp)
    prediction = inception_net.predict(inp).flatten()

    # Apply softmax to convert logits to probabilities
    probabilities = tf.nn.softmax(prediction).numpy()

    # Convert probabilities to percentages and round them off to 2 decimal places
    percentages = [round(p * 100, 5) for p in probabilities]

    confidences = {labels[i]: percentages[i] for i in range(1000)}

    # Sort by confidence, highest confidence first
    confidences = dict(sorted(confidences.items(), key=lambda item: item[1], reverse=True))

    return confidences

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part', 400

    file = request.files['file'].read()
    image = Image.open(io.BytesIO(file)).resize((224, 224))
    image_array = np.array(image)
    result = classify_image(image_array)
    
    limit_str = request.args.get('limit', '5')
    try:
        limit = int(limit_str)
    except ValueError:
        return 'Invalid limit. Limit should be an integer.', 400
    
    # Take the first `limit` items from the already sorted predictions
    top_predictions = dict(list(result.items())[:limit])
    return top_predictions

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)