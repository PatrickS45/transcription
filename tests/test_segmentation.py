from config import FormatConfig, PipelineConfig
from models import Word
from segmentation import HardConstraints, layout_lines, segment_for_format, segment_words
from tests.conftest import make_words


def default_constraints(**overrides) -> HardConstraints:
    params = dict(
        max_words=14,
        max_chars=0,
        duree_min_ms=833,
        duree_max_ms=7000,
        ecart_min_ms=83,
        car_sec_max=20.0,
    )
    params.update(overrides)
    return HardConstraints(**params)


def test_cut_on_pause():
    """Les limites de sous-titres s'alignent sur les silences détectés."""
    words = make_words(["Bonjour", "à", "tous", "et", "bienvenue"], gap=0.05)
    # Grande pause après "tous".
    pause = 1.0
    for w in words[3:]:
        w.start += pause
        w.end += pause
    subs = segment_words(words, default_constraints(), seuil_silence_ms=250)
    assert len(subs) == 2
    assert [w.text for w in subs[0].words] == ["Bonjour", "à", "tous"]
    assert [w.text for w in subs[1].words] == ["et", "bienvenue"]


def test_short_pause_not_cut():
    """Un silence sous le seuil ne provoque pas de coupure."""
    words = make_words(["un", "deux", "trois", "quatre"], gap=0.1)  # 100ms < 250ms
    subs = segment_words(words, default_constraints(), seuil_silence_ms=250)
    assert len(subs) == 1


def test_hard_constraint_forces_cut():
    """Flux continu sans pause : la contrainte de mots max force la coupure."""
    words = make_words([f"mot{i}" for i in range(20)], gap=0.05)
    subs = segment_words(words, default_constraints(max_words=8), seuil_silence_ms=250)
    assert all(len(s.words) <= 8 for s in subs)
    assert sum(len(s.words) for s in subs) == 20


def test_backtrack_to_pause():
    """Quand une contrainte force une coupure, on recule vers la dernière pause
    plutôt que de couper au milieu d'un groupe continu."""
    # Groupe court + pause (300ms) + groupe continu qui déborde la limite.
    words = make_words(["a", "b"], gap=0.05)
    tail = make_words([f"m{i}" for i in range(7)], start=words[-1].end + 0.3, gap=0.05)
    all_words = words + tail
    subs = segment_words(all_words, default_constraints(max_words=6, duree_min_ms=2000),
                         seuil_silence_ms=250)
    # La coupure doit tomber sur la pause après "b", pas au milieu du groupe continu.
    assert [w.text for w in subs[0].words] == ["a", "b"]


def test_duree_max_respected():
    words = make_words([f"mot{i}" for i in range(10)], word_dur=1.0, gap=0.1)
    subs = segment_words(words, default_constraints(), seuil_silence_ms=5000)
    assert all(s.duration_ms <= 7000 for s in subs)


def test_min_duration_extension():
    """Un sous-titre trop court est étendu vers la durée min."""
    words = [Word("Oui", 0.0, 0.2)]
    words2 = make_words(["ensuite", "on", "continue"], start=5.0)
    subs = segment_words(words + words2, default_constraints(), seuil_silence_ms=250)
    assert subs[0].duration_ms >= 833


def test_ecart_min_between_subtitles():
    words = make_words(["premier", "bloc", "de", "mots"], gap=0.05)
    tail = make_words(["second", "bloc", "juste", "après"],
                      start=words[-1].end + 0.3, gap=0.05)
    subs = segment_words(words + tail, default_constraints(), seuil_silence_ms=250)
    for a, b in zip(subs, subs[1:]):
        assert (b.start - a.end) * 1000 >= 82.9


def test_layout_lines_balanced():
    lines = layout_lines(["un", "deux", "trois", "quatre"], max_lines=2,
                         max_words_per_line=3)
    assert lines is not None
    assert len(lines) <= 2
    assert " ".join(lines).split() == ["un", "deux", "trois", "quatre"]


def test_layout_lines_impossible():
    assert layout_lines(["a"] * 10, max_lines=2, max_words_per_line=3) is None


def test_segment_for_format_defaults():
    cfg = PipelineConfig()
    fmt = FormatConfig(mots_max_par_ligne=4, lignes_max=2)
    words = make_words([f"mot{i}" for i in range(30)], gap=0.05)
    subs = segment_for_format(words, cfg, fmt)
    assert all(len(s.words) <= 8 for s in subs)  # 4 mots × 2 lignes
    assert all(len(s.lines) <= 2 for s in subs)
    assert all(all(len(l.split()) <= 4 for l in s.lines) for s in subs)
