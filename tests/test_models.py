from models import Segment, TranscriptionResult, Word


def test_json_roundtrip(tmp_path):
    result = TranscriptionResult(
        segments=[
            Segment(words=[Word("Bonjour", 0.0, 0.4), Word("à", 0.5, 0.6),
                           Word("tous", 0.6, 1.0)]),
            Segment(words=[Word("voici", 2.0, 2.4), Word("l'été", 2.5, 3.0)]),
        ],
        language="fr",
        engine="faster-whisper",
        model="large-v3",
        audio_duration=3.5,
    )
    path = tmp_path / "transcription.json"
    result.save_json(path)
    loaded = TranscriptionResult.load_json(path)
    assert loaded.language == "fr"
    assert loaded.model == "large-v3"
    assert loaded.text == "Bonjour à tous voici l'été"
    assert len(loaded.words) == 5
    assert loaded.words[0].start == 0.0
    assert loaded.words[-1].end == 3.0
    # Les accents doivent survivre au JSON (UTF-8).
    assert "l'été" in path.read_text(encoding="utf-8")


def test_segment_properties():
    seg = Segment(words=[Word("un", 1.0, 1.2), Word("deux", 1.3, 1.6)])
    assert seg.start == 1.0
    assert seg.end == 1.6
    assert seg.text == "un deux"
