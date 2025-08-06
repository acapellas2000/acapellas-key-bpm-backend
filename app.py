from flask import Flask, request, jsonify
import os
import uuid
import librosa
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # To pozwoli na zapytania CORS z frontendu (Twojej strony)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def detect_key(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    if chroma_mean[0] > chroma_mean[7]:
        return "C Major"
    else:
        return "A Minor"

@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        y, sr = librosa.load(filepath, sr=None)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        key = detect_key(y, sr)

        result = {
            "bpm": round(tempo),
            "key": key
        }
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(filepath)

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
