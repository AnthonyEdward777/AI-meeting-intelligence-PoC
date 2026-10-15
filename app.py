
import os
import uuid
from functools import lru_cache
from pathlib import Path

import numpy as np
import ollama
import soundfile as sf
import soxr
import torch
from flask import Flask, jsonify, render_template, request, send_from_directory
from pyannote.audio import Pipeline
from transformers import AutoProcessor, CohereAsrForConditionalGeneration
from transformers.audio_utils import mel_filter_bank
from transformers.models.cohere_asr import feature_extraction_cohere_asr
from werkzeug.utils import secure_filename

from alignment import (
    align_transcript_words,
    assign_speakers_to_words,
    group_words_by_speaker_turn,
)


# --------------------------------------------------
# Flask application and configuration
# --------------------------------------------------

app = Flask(__name__)

PROJECT_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = PROJECT_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

COHERE_MODEL_ID = "CohereLabs/cohere-transcribe-arabic-07-2026"

COHERE_MODEL_PATH = Path(
    os.environ.get(
        "COHERE_MODEL_PATH",
        str(
            PROJECT_DIR
            / "models"
            / "cohere-transcribe-arabic-07-2026"
        ),
    )
)

TRANSCRIPTION_LANGUAGE = "ar"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_DTYPE = (
    torch.bfloat16 if DEVICE == "cuda" else torch.float32
)

SAMPLE_RATE = 16000


# --------------------------------------------------
# Cohere audio feature-extraction compatibility
# --------------------------------------------------

class _NoNumbaLibrosa:
    """
    Cohere's feature extractor needs librosa.filters.mel.
    Use Transformers' mel filter implementation instead.
    """

    class filters:
        @staticmethod
        def mel(sr, n_fft, n_mels, fmin, fmax, norm):
            filters = mel_filter_bank(
                1 + n_fft // 2,
                n_mels,
                fmin,
                fmax,
                sr,
                norm=norm,
                mel_scale="slaney",
            )
            return filters.T.astype(np.float32)


feature_extraction_cohere_asr.librosa = _NoNumbaLibrosa


# --------------------------------------------------
# Audio loading for Cohere
# --------------------------------------------------

def load_audio(audio_path):
    """Load audio as mono, 16 kHz float32 audio."""

    audio, sample_rate = sf.read(
        audio_path,
        dtype="float32",
        always_2d=True,
    )

    # Convert stereo or multichannel audio to mono.
    audio = audio.mean(axis=1)

    # Resample when the input sample rate differs from 16 kHz.
    if sample_rate != SAMPLE_RATE:
        audio = soxr.resample(
            audio,
            sample_rate,
            SAMPLE_RATE,
        )

    return audio


# --------------------------------------------------
# Cohere ASR model
# --------------------------------------------------

@lru_cache(maxsize=1)
def get_transcription_components():
    """Load the Cohere processor and model once."""

    # Without a local copy, load from the Hugging Face cache (still offline).
    model_source = (
        COHERE_MODEL_PATH
        if COHERE_MODEL_PATH.is_dir()
        else COHERE_MODEL_ID
    )

    processor = AutoProcessor.from_pretrained(
    model_source,
    token=True,
    )

    model = CohereAsrForConditionalGeneration.from_pretrained(
    model_source,
    dtype=MODEL_DTYPE,
    token=True,
    )

    model = model.to(DEVICE).eval()

    return processor, model


def transcribe_audio(audio_path):
    """Transcribe the meeting audio using Cohere."""

    processor, model = get_transcription_components()

    audio = load_audio(audio_path)

    inputs = processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        language=TRANSCRIPTION_LANGUAGE,
    )

    audio_chunk_index = inputs.pop("audio_chunk_index", None)

    inputs = inputs.to(
        model.device,
        dtype=model.dtype,
    )

    with torch.inference_mode():
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=448,
        )

    if audio_chunk_index is not None:
        decoded = processor.decode(
            output_tokens,
            skip_special_tokens=True,
            audio_chunk_index=audio_chunk_index,
            language=TRANSCRIPTION_LANGUAGE,
        )
    else:
        decoded = processor.decode(
            output_tokens,
            skip_special_tokens=True,
        )

    if isinstance(decoded, list):
        return " ".join(
            text.strip() for text in decoded
        ).strip()

    return decoded.strip()


# --------------------------------------------------
# Pyannote speaker-diarization model
# --------------------------------------------------

@lru_cache(maxsize=1)
def get_diarization_pipeline():
    """Load the pyannote diarization pipeline once."""

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1"
    )

    if pipeline is None:
        raise RuntimeError(
            "Could not load the pyannote diarization model."
        )

    pipeline.to(torch.device(DEVICE))

    return pipeline


def diarize_audio(audio_path, num_speakers=None):
    """Return detected speaker turns with start/end timestamps.

    num_speakers=None lets pyannote estimate the count itself.
    """

    pipeline = get_diarization_pipeline()

    waveform, sample_rate = sf.read(
        audio_path,
        dtype="float32",
        always_2d=True,
    )

    # Convert from (samples, channels) to (channels, samples).
    waveform = waveform.T

    audio_data = {
        "waveform": torch.from_numpy(waveform.copy()),
        "sample_rate": sample_rate,
    }

    diarization = pipeline(audio_data, num_speakers=num_speakers)

    # Support the pyannote.audio 4.x result format.
    annotation = getattr(
        diarization,
        "speaker_diarization",
        diarization,
    )

    segments = []

    for turn, _, speaker in annotation.itertracks(
        yield_label=True
    ):
        segments.append({
            "start": round(float(turn.start), 2),
            "end": round(float(turn.end), 2),
            "speaker": str(speaker),
        })

    return segments


# --------------------------------------------------
# Flask pages and JavaScript
# --------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/script.js")
def script():
    return send_from_directory(
        str(PROJECT_DIR / "templates"),
        "script.js",
    )


# --------------------------------------------------
# Main meeting-analysis endpoint
# --------------------------------------------------

@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("audio")

    if not file or not file.filename:
        return jsonify({
            "error": "No audio file uploaded."
        }), 400

    # Optional speaker count; blank means let pyannote estimate it.
    num_speakers = request.form.get("num_speakers", "").strip()

    if num_speakers:
        if not num_speakers.isdigit() or not 1 <= int(num_speakers) <= 20:
            return jsonify({
                "error": "Number of speakers must be between 1 and 20."
            }), 400
        num_speakers = int(num_speakers)
    else:
        num_speakers = None

    # Generate a unique filename for the uploaded recording.
    extension = Path(
        secure_filename(file.filename)
    ).suffix

    filepath = UPLOAD_DIR / f"{uuid.uuid4().hex}{extension}"
    file.save(filepath)

    try:
        # ------------------------------------------
        # Step 1: Transcribe with Cohere
        # ------------------------------------------

        transcript = transcribe_audio(filepath)

        # ------------------------------------------
        # Step 2: Detect speaker turns with pyannote
        # ------------------------------------------

        speaker_segments = diarize_audio(filepath, num_speakers)

        # ------------------------------------------
        # Step 3: Generate the meeting summary with Qwen
        # ------------------------------------------

        prompt = f"""
        Read this Egyptian Arabic meeting transcript:
        {transcript}

        Write a short summary of the meeting. Reply with the summary only.
        """

        # Only the summary comes from the LLM: asking it to rewrite the
        # whole transcript made long recordings loop into truncated JSON.
        response = ollama.chat(
            model="qwen2.5:7b-instruct",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            options={"num_predict": 1024},
        )

        result = {
            "summary": response["message"]["content"].strip(),
            # Fallback display when word alignment fails.
            "transcript": [
                {"speaker": "UNKNOWN", "text": transcript}
            ],
        }

        # ------------------------------------------
        # Step 4: Align transcript words with audio
        # ------------------------------------------

        try:
            alignment_result = align_transcript_words(
                filepath,
                transcript,
            )

            # --------------------------------------
            # Step 5: Match words to speaker turns
            # --------------------------------------

            aligned_words = assign_speakers_to_words(
                alignment_result["words"],
                speaker_segments,
            )

            # --------------------------------------
            # Step 6: Group words by speaker turn
            # --------------------------------------

            result["speaker_transcript"] = (
                group_words_by_speaker_turn(
                    aligned_words,
                    speaker_segments,
                )
            )

            # Keep word-level results available for validation.
            result["word_alignment"] = (
                alignment_result["words"]
            )

            if alignment_result["skipped_words"]:
                result["alignment_warning"] = (
                    "Some transcript words could not be normalized: "
                    + ", ".join(
                        alignment_result["skipped_words"]
                    )
                )

        except Exception as error:
            # Alignment failure should not discard the summary.
            app.logger.exception(
                "Word alignment failed"
            )

            result["alignment_error"] = (
                f"{type(error).__name__}: {error}"
            )

            result["speaker_transcript"] = []

        # Preserve the original pyannote segments.
        result["speaker_segments"] = speaker_segments

        return jsonify(result)

    except Exception as error:
        # Report other analysis errors to the frontend.
        app.logger.exception(
            "Meeting analysis failed"
        )

        return jsonify({
            "error": f"{type(error).__name__}: {error}"
        }), 500

    finally:
        # Delete the temporary uploaded recording.
        filepath.unlink(missing_ok=True)


# --------------------------------------------------
# Run the Flask application
# --------------------------------------------------

if __name__ == "__main__":
    app.run(port=5000)
