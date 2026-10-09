import json
import os
import sys
import uuid
from functools import lru_cache
from pathlib import Path

# Windows Application Control blocks scikit-learn's DLLs on this machine, and
# transformers imports it if installed. Hiding it lets transformers load.
sys.modules["sklearn"] = None

import numpy as np
import ollama
import soundfile as sf
import soxr
import torch
from flask import Flask, jsonify, render_template, request, send_from_directory
from transformers import AutoProcessor, CohereAsrForConditionalGeneration
from transformers.audio_utils import mel_filter_bank
from transformers.models.cohere_asr import feature_extraction_cohere_asr
from werkzeug.utils import secure_filename


app = Flask(__name__)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
COHERE_MODEL_PATH = Path(
    os.environ.get(
        "COHERE_MODEL_PATH",
        r"C:\Users\Zein\Documents\Mawjood\models\hf\hub\models--CohereLabs--cohere-transcribe-arabic-07-2026\snapshots\c3e911b42149bf7a1e53d5cef9878aee87515a23",
    )
)
TRANSCRIPTION_LANGUAGE = "ar"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
SAMPLE_RATE = 16000


class _NoNumbaLibrosa:
    """Cohere's feature extractor only needs librosa.filters.mel, but librosa
    needs numba, which is also blocked here. Use transformers' own version."""

    class filters:
        @staticmethod
        def mel(sr, n_fft, n_mels, fmin, fmax, norm):
            filters = mel_filter_bank(
                1 + n_fft // 2, n_mels, fmin, fmax, sr, norm=norm, mel_scale="slaney"
            )
            return filters.T.astype(np.float32)


feature_extraction_cohere_asr.librosa = _NoNumbaLibrosa


def load_audio(audio_path):
    audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)  # mono
    if sample_rate != SAMPLE_RATE:
        audio = soxr.resample(audio, sample_rate, SAMPLE_RATE)
    return audio


@lru_cache(maxsize=1)
def get_transcription_components():
    if not COHERE_MODEL_PATH.is_dir():
        raise FileNotFoundError(
            f"Cohere model was not found at {COHERE_MODEL_PATH}. "
            "Set COHERE_MODEL_PATH to the downloaded model directory."
        )

    processor = AutoProcessor.from_pretrained(
        COHERE_MODEL_PATH, local_files_only=True
    )
    model = CohereAsrForConditionalGeneration.from_pretrained(
        COHERE_MODEL_PATH, dtype=MODEL_DTYPE, local_files_only=True
    )
    return processor, model.to(DEVICE).eval()


def transcribe_audio(audio_path):
    processor, model = get_transcription_components()
    audio = load_audio(audio_path)
    inputs = processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        language=TRANSCRIPTION_LANGUAGE,
    )
    audio_chunk_index = inputs.pop("audio_chunk_index", None)
    inputs = inputs.to(model.device, dtype=model.dtype)

    with torch.inference_mode():
        output_tokens = model.generate(**inputs, max_new_tokens=448)

    if audio_chunk_index is not None:
        decoded = processor.decode(
            output_tokens,
            skip_special_tokens=True,
            audio_chunk_index=audio_chunk_index,
            language=TRANSCRIPTION_LANGUAGE,
        )
    else:
        decoded = processor.decode(output_tokens, skip_special_tokens=True)

    if isinstance(decoded, list):
        return " ".join(text.strip() for text in decoded).strip()
    return decoded.strip()


@app.route("/")
def index():
    return render_template("index.html")


# index.html loads "script.js" from the site root, but it lives in templates/
@app.route("/script.js")
def script():
    return send_from_directory("templates", "script.js")


@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("audio")
    if not file or not file.filename:
        return jsonify({"error": "No audio file uploaded."}), 400

    # Random name so uploads can't escape UPLOAD_DIR or overwrite each other
    extension = Path(secure_filename(file.filename)).suffix
    filepath = UPLOAD_DIR / f"{uuid.uuid4().hex}{extension}"
    file.save(filepath)

    try:
        transcript = transcribe_audio(filepath)
        raw_text = f"[UNKNOWN]: {transcript}\n"

        prompt = f"""
        Read this Egyptian Arabic meeting transcript:
        {raw_text}

        1. Figure out who each speaker is based on what they call each other.
        2. Write a summary of the meeting.

        Return EXACTLY this JSON format:
        {{
            "summary": "Meeting summary...",
            "transcript": [
                {{"speaker": "Name", "text": "What they said..."}}
            ]
        }}
        """

        response = ollama.chat(
            model="qwen2.5:7b-instruct",
            messages=[{"role": "user", "content": prompt}],
            format="json",
        )
        result = json.loads(response["message"]["content"])
    finally:
        filepath.unlink(missing_ok=True)

    return jsonify(result)


if __name__ == "__main__":
    app.run(port=5000)
