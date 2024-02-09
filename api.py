from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import requests

app = Flask(__name__)

face_classifier = cv2.CascadeClassifier(r'./haarcascade_frontalface_default.xml')
classifier = load_model('./first-model.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def detect_emotion(image):
    if isinstance(image, str):  # If the image is a file path
        frame = cv2.imread(image)
    elif isinstance(image, np.ndarray):  # If the image is already loaded
        frame = image
    else:
        print("Error: Unsupported image format.")
        return None

    if frame is None:
        print(f"Error: Unable to load image.")
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            return label

    return None

@app.route('/hello', methods=['GET'])
def hello():
    return jsonify({'message': 'Hello, World!'})

@app.route('/image/emotion', methods=['POST'])
def analyze_emotion():
    if 'imageUrl' not in request.json:
        return jsonify({'error': 'imageUrl is missing in request body'}), 400

    image_url = request.json['imageUrl']
    response = requests.get(image_url)

    if response.status_code != 200:
        return jsonify({'error': 'Failed to fetch image from the provided URL'}), 400

    image_content = np.asarray(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(image_content, -1)

    if image is None:
        return jsonify({'error': 'Failed to decode the image'}), 400

    emotion_label = detect_emotion(image)

    if emotion_label:
        return jsonify({'emotion': emotion_label}), 200
    else:
        return jsonify({'error': 'No face detected or unable to determine emotion'}), 400

if __name__ == '__main__':
    app.run(debug=True)
