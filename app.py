from flask import Flask, request, jsonify
import os
import uuid
import librosa
import soundfile as sf
import time
from collections import defaultdict
from datetime import datetime
import logging

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
LOG_FILE = "analysis_log.txt"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Rate-limiting: pamiętamy ostatnią analizę dla IP
last_analysis_time = defaultdict(float)
RATE_LIMIT_SECONDS = 30  # jedno zapytanie na 30s/IP

# Konfiguracja logowania
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format="%(asctime)s - %(message)s")

def detect_key_full(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)

    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F',
                     'F#', 'G', 'G#', 'A', 'A#', 'B']

    major_profiles = librosa.feature.chroma_cqt(
        y=librosa.tone_to_chroma(librosa.key_to_tone("C:maj"), sr=sr), sr=sr).mean(axis=1)
    minor_profiles = librosa.feature.chroma_cqt(
        y=librosa.tone_to_chroma(librosa.key_to_tone("A:min"), sr=sr), sr=sr).mean(axis=1)

    correlations = []
    for i in range(12):
        major_score = sum(chroma_mean * major_profiles[-i:] + major_profiles[:-i])
        minor_score = sum(chroma_mean * minor_profiles[-i:] + minor_profiles[:-i])
        correlations.append((major_score, minor_score))

    best = max(enumerate(correlations), key=lambda x: max(x[1]))[0]
    if correlations[best][0] > correlations[best][1]:
        return f"{pitch_classes[best]} Major"
    else:
        return f"{pitch_classes[best]} Minor"

@app.before_request
def rate_limit():
    if request.endpoint == "analyze":
        ip = request.remote_addr
        now = time.time()
        if now - last_analysis_time[ip] < RATE_LIMIT_SECONDS:
            return jsonify({"error": f"Too many requests. Try again in {int(RATE_LIMIT_SECONDS - (now - last_analysis_time[ip]))}s"}), 429
        last_analysis_time[ip] = now

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
        duration = librosa.get_duration(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        key = detect_key_full(y, sr)

        # Dane pliku audio
        audio_info = sf.info(filepath)
        bitrate = (os.path.getsize(filepath) * 8) / duration if duration > 0 else 0

        result = {
            "bpm": round(tempo),
            "key": key,
            "duration_sec": round(duration, 2),
            "format": audio_info.format,
            "bitrate_kbps": round(bitrate / 1000, 2)
        }

        # Logowanie
        log_entry = f"{request.remote_addr} - {file.filename} - {result}"
        logging.info(log_entry)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(filepath)

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
