from flask import Flask, request, jsonify
from deepface import DeepFace

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


if __name__ == '__main__':
    app.run(debug=True, port=5001)
