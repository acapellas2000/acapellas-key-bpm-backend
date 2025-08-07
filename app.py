from flask import Flask, request, jsonify
import os
import uuid
import librosa
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["*"])  # Pozwala na wszystkie origins - w produkcji ustaw konkretny URL

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def detect_key(y, sr):
    """Prosta detekcja tonacji - można ulepszyć"""
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = chroma.mean(axis=1)
        
        # Prosty algorytm - można zastąpić bardziej zaawansowanym
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key_idx = chroma_mean.argmax()
        
        # Określenie czy major czy minor (uproszczone)
        if chroma_mean[key_idx] > chroma_mean[(key_idx + 9) % 12]:  # sprawdzanie major vs relative minor
            return f"{keys[key_idx]} Major"
        else:
            return f"{keys[(key_idx + 9) % 12]} Minor"
    except Exception as e:
        return "Unknown"

def get_audio_info(filepath, y, sr):
    """Pobiera dodatkowe informacje o pliku audio"""
    try:
        duration = librosa.get_duration(y=y, sr=sr)
        file_size = os.path.getsize(filepath)
        
        # Próba określenia formatu na podstawie rozszerzenia
        file_ext = os.path.splitext(filepath)[1].lower()
        format_map = {'.mp3': 'MP3', '.wav': 'WAV', '.flac': 'FLAC', '.m4a': 'M4A'}
        audio_format = format_map.get(file_ext, 'Unknown')
        
        # Przybliżone obliczenie bitrate (uproszczone)
        if duration > 0:
            bitrate = (file_size * 8) / (duration * 1000)  # kbps
        else:
            bitrate = 0
            
        return {
            'duration_sec': round(duration, 2),
            'format': audio_format,
            'bitrate_kbps': round(bitrate)
        }
    except Exception as e:
        return {
            'duration_sec': 0,
            'format': 'Unknown',
            'bitrate_kbps': 0
        }

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        # Sprawdzenie czy plik został wysłany
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400
            
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        # Sprawdzenie rozszerzenia pliku
        allowed_extensions = ['.mp3', '.wav', '.flac', '.m4a']
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return jsonify({"error": "Unsupported file format"}), 400

        # Zapisanie pliku
        filename = str(uuid.uuid4()) + file_ext
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        try:
            # Analiza audio
            print(f"Loading file: {filepath}")
            y, sr = librosa.load(filepath, sr=None)
            print(f"Audio loaded successfully. Sample rate: {sr}")
            
            # Detekcja tempa (BPM)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            print(f"Tempo detected: {tempo}")
            
            # Detekcja tonacji
            key = detect_key(y, sr)
            print(f"Key detected: {key}")
            
            # Dodatkowe informacje
            audio_info = get_audio_info(filepath, y, sr)
            
            result = {
                "bpm": round(float(tempo)),
                "key": key,
                "duration_sec": audio_info['duration_sec'],
                "format": audio_info['format'],
                "bitrate_kbps": audio_info['bitrate_kbps']
            }
            
            print(f"Analysis complete: {result}")
            return jsonify(result)
            
        except Exception as e:
            print(f"Analysis error: {str(e)}")
            return jsonify({"error": f"Analysis failed: {str(e)}"}), 500
            
        finally:
            # Usuń plik po analizie
            if os.path.exists(filepath):
                os.remove(filepath)
                
    except Exception as e:
        print(f"General error: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "Backend is running", "message": "Key & BPM Analyzer API"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)