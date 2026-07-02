from models import Segment, TranscriptionResult, Word, merge_elision_fragments


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


def test_merge_elision_fragments():
    """Les fragments tokenisés séparément de part et d'autre d'une apostrophe
    ou d'un trait d'union doivent être recollés sans espace parasite."""
    seg = Segment(words=[
        Word("d", 0.0, 0.1), Word("'associé", 0.1, 0.4),
        Word("c", 0.5, 0.6), Word("'est", 0.6, 0.8),
        Word("-à", 0.8, 0.9), Word("-dire", 0.9, 1.1),
        Word("une", 1.2, 1.4),
    ])
    result = TranscriptionResult(segments=[seg])
    merged = merge_elision_fragments(result)
    words = merged.segments[0].words
    assert [w.text for w in words] == ["d'associé", "c'est-à-dire", "une"]
    # Les timestamps du premier et dernier fragment fusionné sont préservés.
    assert words[0].start == 0.0 and words[0].end == 0.4
    assert words[1].start == 0.5 and words[1].end == 1.1


def test_merge_elision_leading_fragment_kept_as_is():
    """Un fragment en tête de segment (sans mot précédent) reste tel quel."""
    seg = Segment(words=[Word("'est", 0.0, 0.2), Word("un test", 0.3, 0.6)])
    merged = merge_elision_fragments(TranscriptionResult(segments=[seg]))
    assert [w.text for w in merged.segments[0].words] == ["'est", "un test"]
