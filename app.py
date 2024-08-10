from flask import Flask, render_template, request, jsonify
import os
import base64

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

    # Decode the image data
    image_data = image_data.split(',')[1]
    image_data = base64.b64decode(image_data)

    # Save the image to a file
    image_path = os.path.join('saved_images', 'captured_image.jpg')
    with open(image_path, 'wb') as f:
        f.write(image_data)

    return jsonify({"message": "Image saved successfully", "image_path": image_path})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1000)
