import os
import cv2
import requests
from flask import Flask, request, jsonify
from deepface import DeepFace
import numpy as np

app = Flask(__name__)


def get_image_from_url(image_url):
    try:
        response = requests.get(image_url)
        img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Unable to load image from {image_url}")
        return img

    except Exception as e:
        print(f"Error downloading image from URL: {e}")
        raise ValueError(f"Error downloading image: {str(e)}")


@app.route('/extract-embedding', methods=['POST'])
def extract_embedding():
    data = request.json
    image_url = data['image_url']

    try:
        img = get_image_from_url(image_url)
        embedding = DeepFace.represent(img_path=img, model_name="VGG-Face")[0]["embedding"]
        return jsonify({'embedding': embedding}), 200

    except Exception as e:
        print(f"Error extracting face embedding: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/detect-face', methods=['POST'])
def detect_face():
    data = request.json
    image_url = data['image_url']

    try:
        img = get_image_from_url(image_url)
        face_objs = DeepFace.extract_faces(img, detector_backend='opencv', enforce_detection=False)
        face_count = len(face_objs) if isinstance(face_objs, list) else 0
        return jsonify({'face_count': face_count}), 200

    except Exception as e:
        print(f"Error detecting face: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
