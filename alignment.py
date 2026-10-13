
import re
from functools import lru_cache

import numpy as np
import soundfile as sf
import soxr
import torch
import uroman as ur
from torchaudio.pipelines import MMS_FA


SAMPLE_RATE = MMS_FA.sample_rate


@lru_cache(maxsize=1)
def get_alignment_components():
    """Load the alignment model once and reuse it."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MMS_FA.get_model(with_star=False)
    model = model.to(device).eval()

    tokenizer = MMS_FA.get_tokenizer()
    aligner = MMS_FA.get_aligner()

    return model, tokenizer, aligner, device


@lru_cache(maxsize=1)
def get_romanizer():
    """Reuse the Uroman text processor."""
    return ur.Uroman()


def align_transcript_words(audio_path, transcript):
    """
    Estimate start/end timestamps for transcript words.

    Arabic text is romanized for alignment, but original words are
    retained for display. This function does not transcribe audio.
    """
    if not transcript or not transcript.strip():
        raise ValueError("The transcript is empty.")

    romanizer = get_romanizer()

    original_words = transcript.split()
    display_words = []
    normalized_words = []
    skipped_words = []

    # Normalize each source word separately to preserve its display text.
    for original_word in original_words:
        romanized = romanizer.romanize_string(
            original_word,
            lcode="ara"
        )

        normalized = re.sub(
            r"[^a-z']",
            " ",
            romanized.lower()
        )
        normalized_parts = normalized.split()

        if not normalized_parts:
            skipped_words.append(original_word)
            continue

        for index, normalized_word in enumerate(normalized_parts):
            normalized_words.append(normalized_word)

            # Normally there is one normalized token per source word.
            # Preserve the original text for the first token.
            display_words.append(
                original_word if index == 0 else normalized_word
            )

    if not normalized_words:
        raise ValueError(
            "No supported words remained after transcript normalization."
        )

    # Read the audio and convert it to mono, 16 kHz float audio.
    audio, sample_rate = sf.read(
        audio_path,
        dtype="float32",
        always_2d=True
    )
    audio = audio.mean(axis=1)

    if sample_rate != SAMPLE_RATE:
        audio = soxr.resample(
            audio,
            sample_rate,
            SAMPLE_RATE
        )

    if audio.size == 0:
        raise ValueError("The audio file is empty.")

    waveform = torch.from_numpy(
        np.asarray(audio, dtype=np.float32).copy()
    ).unsqueeze(0)

    model, tokenizer, aligner, device = get_alignment_components()

    # Generate acoustic emissions and align the known transcript.
    with torch.inference_mode():
        emissions, _ = model(waveform.to(device))
        token_spans = aligner(
            emissions[0],
            tokenizer(normalized_words)
        )

    if len(token_spans) != len(normalized_words):
        raise RuntimeError(
            "The aligner returned a different number of words "
            "than expected."
        )

    # Convert model frame indexes to seconds.
    seconds_per_frame = (
        waveform.shape[1] / emissions.shape[1] / SAMPLE_RATE
    )
    audio_duration = waveform.shape[1] / SAMPLE_RATE

    aligned_words = []

    for original_word, normalized_word, spans in zip(
        display_words,
        normalized_words,
        token_spans
    ):
        if not spans:
            continue

        start = max(0.0, spans[0].start * seconds_per_frame)
        end = min(
            audio_duration,
            spans[-1].end * seconds_per_frame
        )

        # Duration-weighted alignment score.
        total_weight = sum(
            max(1, span.end - span.start)
            for span in spans
        )
        score = sum(
            float(span.score) * max(1, span.end - span.start)
            for span in spans
        ) / total_weight

        aligned_words.append({
            "text": original_word,
            "normalized": normalized_word,
            "start": round(start, 2),
            "end": round(end, 2),
            "alignment_score": round(score, 3)
        })

    return {
        "words": aligned_words,
        "skipped_words": skipped_words
    }


def assign_speakers_to_words(aligned_words, speaker_segments):
    """Assign each aligned word to the speaker turn with the most overlap."""
    result = []

    for word in aligned_words:
        word_start = word["start"]
        word_end = word["end"]
        duration = max(0.001, word_end - word_start)

        best_index = None
        best_overlap = 0.0

        for index, segment in enumerate(speaker_segments):
            overlap = max(
                0.0,
                min(word_end, segment["end"])
                - max(word_start, segment["start"])
            )

            if overlap > best_overlap:
                best_overlap = overlap
                best_index = index

        # Avoid assigning a speaker when almost none of the word
        # overlaps that speaker's detected turn.
        if (
            best_index is not None
            and best_overlap / duration >= 0.20
        ):
            speaker = speaker_segments[best_index]["speaker"]
        else:
            best_index = None
            speaker = "UNKNOWN"

        result.append({
            **word,
            "speaker": speaker,
            "_speaker_segment_index": best_index
        })

    return result


def group_words_by_speaker_turn(aligned_words, speaker_segments):
    """Build one Arabic text segment for each detected speaker turn."""
    grouped = []

    for index, segment in enumerate(speaker_segments):
        words = [
            word
            for word in aligned_words
            if word["_speaker_segment_index"] == index
        ]

        if not words:
            continue

        grouped.append({
            "speaker": segment["speaker"],
            "start": segment["start"],
            "end": segment["end"],
            "text": " ".join(word["text"] for word in words),
            "words": [
                {
                    "text": word["text"],
                    "start": word["start"],
                    "end": word["end"],
                    "alignment_score": word["alignment_score"]
                }
                for word in words
            ]
        })

    return grouped
