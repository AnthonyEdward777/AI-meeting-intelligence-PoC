import os
import json
import whisperx
import ollama
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
os.makedirs('uploads', exist_ok=True)

device = "cpu" # Change to "cuda" if you have an NVIDIA GPU and to "cpu" if no Nvidia GPU.
hf_token = "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"

print("Loading Whisper & Diarization models...")
asr_model = whisperx.load_model("large-v3", device, compute_type="float16")
diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=hf_token, device=device)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['audio']
    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)

    audio = whisperx.load_audio(filepath)
    result = asr_model.transcribe(audio, batch_size=4)

    align_model, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], align_model, metadata, audio, device, return_char_alignments=False)

    diarize_segments = diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    raw_text = ""
    for segment in result["segments"]:
        speaker = segment.get("speaker", "UNKNOWN")
        raw_text += f"[{speaker}]: {segment['text']}\n"

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
        model='qwen2.5:7b',
        messages=[{'role': 'user', 'content': prompt}],
        format='json'
    )
    
    os.remove(filepath)
    return jsonify(json.loads(response['message']['content']))

if __name__ == '__main__':
    app.run(debug=True, port=5000)