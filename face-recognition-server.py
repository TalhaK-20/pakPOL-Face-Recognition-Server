from flask import Flask, request, jsonify
from deepface import DeepFace
import cv2

app = Flask(__name__)


@app.route('/extract-embedding', methods=['POST'])
def extract_embedding():
    data = request.json
    image_path = data['image_path']

    try:
        embedding = DeepFace.represent(img_path=image_path, model_name="VGG-Face")[0]["embedding"]
        return jsonify({'embedding': embedding}), 200

    except Exception as e:
        print(f"Error extracting face embedding: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/detect-face', methods=['POST'])
def detect_face():
    data = request.json
    image_path = data['image_path']

    try:
        img = cv2.imread(image_path)
        face_objs = DeepFace.extract_faces(img, detector_backend='opencv', enforce_detection=False)
        face_count = len(face_objs) if isinstance(face_objs, list) else 0
        return jsonify({'face_count': face_count}), 200

    except Exception as e:
        print(f"Error detecting face: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5001)
