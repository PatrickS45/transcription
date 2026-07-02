from config import CorrectionConfig
from correction import build_prompt, correct_transcription, realign_words
from models import Segment, TranscriptionResult, Word
from tests.conftest import make_words


def test_realign_same_word_count_keeps_timestamps():
    original = make_words(["bonjour", "avous", "tous"])
    realigned = realign_words(original, "Bonjour à-vous tous")
    assert [w.text for w in realigned] == ["Bonjour", "à-vous", "tous"]
    for orig, new in zip(original, realigned):
        assert new.start == orig.start
        assert new.end == orig.end


def test_realign_split_words_preserves_bounds():
    """Si le LLM scinde des mots, l'alignement est reconstruit de façon
    déterministe et les bornes du segment sont préservées."""
    original = make_words(["bonjouratous", "lesamis"])
    realigned = realign_words(original, "bonjour à tous les amis")
    assert len(realigned) == 5
    assert realigned[0].start == original[0].start
    assert realigned[-1].end == original[-1].end
    for a, b in zip(realigned, realigned[1:]):
        assert a.start <= a.end <= b.start + 1e-9


def test_realign_merge_words_preserves_bounds():
    original = make_words(["au", "jour", "d'hui", "même"])
    realigned = realign_words(original, "aujourd'hui même")
    assert len(realigned) == 2
    assert realigned[0].start == original[0].start
    assert realigned[-1].end == original[-1].end


def test_realign_is_deterministic():
    original = make_words(["un", "deux", "trois"])
    a = realign_words(original, "un deux trois quatre")
    b = realign_words(original, "un deux trois quatre")
    assert [(w.text, w.start, w.end) for w in a] == [(w.text, w.start, w.end) for w in b]


def test_correction_disabled_is_identity():
    """Hors chemin critique : pipeline identique quand l'étape est désactivée."""
    result = TranscriptionResult(segments=[Segment(words=make_words(["salut"]))])
    out = correct_transcription(result, CorrectionConfig(active=False))
    assert out is result


def test_correction_never_touches_timestamps(monkeypatch):
    """Le LLM ne touche jamais aux timestamps du segment."""
    import correction as corr_mod

    result = TranscriptionResult(
        segments=[Segment(words=make_words(["monsieur", "dupont", "arrive"]))]
    )
    monkeypatch.setitem(
        corr_mod._BACKENDS, "ollama",
        lambda text, cfg: "Monsieur Dupont arrive.",
    )
    out = correct_transcription(result, CorrectionConfig(active=True, backend="ollama"))
    assert out.segments[0].text == "Monsieur Dupont arrive."
    assert out.segments[0].start == result.segments[0].start
    assert out.segments[0].end == result.segments[0].end


def test_glossary_in_prompt(tmp_path):
    glossary_file = tmp_path / "glossaire.txt"
    glossary_file.write_text("Loiret\nOrléans Métropole\n", encoding="utf-8")
    from correction import load_glossary

    terms = load_glossary(str(glossary_file))
    prompt = build_prompt("le loiret et orleans metropole", terms)
    assert "Loiret" in prompt
    assert "Orléans Métropole" in prompt
    assert "le loiret et orleans metropole" in prompt
