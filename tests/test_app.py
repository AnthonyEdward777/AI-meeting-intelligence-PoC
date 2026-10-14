import importlib
import io
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


sys.modules["ollama"] = MagicMock()
sys.modules["torch"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["transformers.audio_utils"] = MagicMock()
sys.modules["transformers.models.cohere_asr"] = MagicMock()
sys.modules["pyannote.audio"] = MagicMock()
sys.modules["torchaudio.pipelines"] = MagicMock()

app = importlib.import_module("app")


class FakeInputs(dict):
    def to(self, *_args, **_kwargs):
        return self


class CohereTranscriptionTests(unittest.TestCase):
    def test_transcribe_audio_returns_decoded_local_cohere_text(self):
        self.assertTrue(hasattr(app, "transcribe_audio"))

        processor = MagicMock()
        processor.return_value = FakeInputs(input_features="audio-features")
        processor.decode.return_value = "هذا اجتماع تجريبي"
        model = SimpleNamespace(device="cpu", dtype="float32", generate=MagicMock(return_value="tokens"))

        with (
            patch.object(app, "get_transcription_components", return_value=(processor, model)),
            patch.object(app, "load_audio", return_value=[0.1, 0.2]),
        ):
            transcription = app.transcribe_audio("meeting.wav")

        self.assertEqual(transcription, "هذا اجتماع تجريبي")
        processor.assert_called_once_with(
            [0.1, 0.2], sampling_rate=16000, return_tensors="pt", language="ar"
        )
        model.generate.assert_called_once_with(
            input_features="audio-features", max_new_tokens=448
        )


class AnalyzeRouteTests(unittest.TestCase):
    def test_upload_is_saved_inside_upload_dir_and_removed_on_error(self):
        saved_paths = []

        def fake_transcribe(path):
            saved_paths.append(path)
            self.assertTrue(path.exists())
            raise RuntimeError("transcription failed")

        client = app.app.test_client()
        with patch.object(app, "transcribe_audio", side_effect=fake_transcribe):
            response = client.post(
                "/analyze",
                data={"audio": (io.BytesIO(b"fake audio"), "../../evil.wav")},
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(saved_paths[0].parent, app.UPLOAD_DIR)
        self.assertEqual(saved_paths[0].suffix, ".wav")
        self.assertFalse(saved_paths[0].exists())

    def test_missing_file_returns_400(self):
        response = app.app.test_client().post("/analyze", data={})
        self.assertEqual(response.status_code, 400)

    def test_invalid_speaker_count_returns_400(self):
        for value in ("0", "abc", "-2", "21"):
            response = app.app.test_client().post(
                "/analyze",
                data={
                    "audio": (io.BytesIO(b"fake audio"), "a.wav"),
                    "num_speakers": value,
                },
            )
            self.assertEqual(response.status_code, 400, value)

    def test_speaker_count_is_passed_to_diarization(self):
        with (
            patch.object(app, "transcribe_audio", return_value="x"),
            patch.object(app, "diarize_audio", side_effect=RuntimeError) as diarize,
        ):
            app.app.test_client().post(
                "/analyze",
                data={
                    "audio": (io.BytesIO(b"fake audio"), "a.wav"),
                    "num_speakers": "2",
                },
            )

        self.assertEqual(diarize.call_args.args[1], 2)


class SpellOutNumbersTests(unittest.TestCase):
    def test_numbers_become_arabic_words(self):
        from alignment import spell_out_numbers

        self.assertEqual(spell_out_numbers("57"), "سبعة و خمسون")
        self.assertEqual(spell_out_numbers("كلمة"), "كلمة")
        # Phone-style numbers are read digit by digit.
        self.assertEqual(spell_out_numbers("012"), "صفر واحد اثنان")


if __name__ == "__main__":
    unittest.main()
