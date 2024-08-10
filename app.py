from flask import Flask, render_template, request, jsonify
import os
import base64
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import cv2



app = Flask(__name__, static_url_path='/static')

# Ensure the folder exists to save images
if not os.path.exists('saved_images'):
    os.makedirs('saved_images')

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/save-image', methods=['POST'])
def save_image():
    data = request.get_json()
    image_data = data['image']
    plant_diseases = [
    "Apple - Apple Scab",
    "Apple - Black Rot",
    "Apple - Cedar Apple Rust",
    "Apple - Healthy",
    "Blueberry - Healthy",
    "Cherry (including sour) - Powdery Mildew",
    "Cherry (including sour) - Healthy",
    "Corn (maize) - Cercospora Leaf Spot / Gray Leaf Spot",
    "Corn (maize) - Common Rust",
    "Corn (maize) - Northern Leaf Blight",
    "Corn (maize) - Healthy",
    "Grape - Black Rot",
    "Grape - Esca (Black Measles)",
    "Grape - Leaf Blight (Isariopsis Leaf Spot)",
    "Grape - Healthy",
    "Orange - Haunglongbing (Citrus Greening)",
    "Peach - Bacterial Spot",
    "Peach - Healthy",
    "Pepper (bell) - Bacterial Spot",
    "Pepper (bell) - Healthy",
    "Potato - Early Blight",
    "Potato - Late Blight",
    "Potato - Healthy",
    "Raspberry - Healthy",
    "Soybean - Healthy",
    "Squash - Powdery Mildew",
    "Strawberry - Leaf Scorch",
    "Strawberry - Healthy",
    "Tomato - Bacterial Spot",
    "Tomato - Early Blight",
    "Tomato - Late Blight",
    "Tomato - Leaf Mold",
    "Tomato - Septoria Leaf Spot",
    "Tomato - Spider Mites / Two-Spotted Spider Mite",
    "Tomato - Target Spot",
    "Tomato - Tomato Yellow Leaf Curl Virus",
    "Tomato - Tomato Mosaic Virus",
    "Tomato - Healthy"
]

    # Decode the image data
    image_data = image_data.split(',')[1]
    image_data = base64.b64decode(image_data)

    # Save the image to a file
    image_path = os.path.join('saved_images', 'captured_image.jpg')
    with open(image_path, 'wb') as f:
        f.write(image_data)

    # Load the model
    model = load_model("model.h5")

    # Preprocess the image
    img_ = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR)
    img_ = cv2.resize(img_, (256, 256))
    img_array = img_to_array(img_)
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    predicted_class = plant_diseases[predicted_class]
    return jsonify({
        "message": "Image saved successfully",
        "image_path": image_path,
        "predicted_class": (predicted_class)
    })   

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1000)
