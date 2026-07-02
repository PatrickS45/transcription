from config import FormatConfig, PipelineConfig
from models import Subtitle, Word
from rendering import FontMeasurer, validate_and_reflow
from segmentation import segment_for_format
from tests.conftest import make_words


def make_cfg(font_path: str) -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.police = font_path
    return cfg


def test_measurer_widths_differ(font_path):
    """La largeur réelle dépend du caractère : un 'i' et un 'm' n'occupent pas
    le même espace (CDC §4)."""
    m = FontMeasurer(font_path, 100)
    assert m.width("iiii") < m.width("mmmm")
    assert m.width("") == 0.0


def test_measurer_scales_with_size(font_path):
    small = FontMeasurer(font_path, 50).width("Bonjour à tous")
    large = FontMeasurer(font_path, 100).width("Bonjour à tous")
    assert large > small * 1.8


def test_all_lines_fit_after_reflow(font_path):
    """Après validation, chaque ligne tient réellement dans le cadre, marge
    de sécurité incluse."""
    cfg = make_cfg(font_path)
    fmt = FormatConfig(taille_police=100, mots_max_par_ligne=7,
                       largeur_sequence_px=1920)
    words = make_words(
        ("la validation par rendu réel garantit que chaque ligne tient "
         "dans le cadre du moniteur programme de Premiere sans déborder "
         "visuellement quelle que soit la taille de police du projet").split(),
        word_dur=0.45, gap=0.15,  # débit réaliste, compatible 20 car/s
    )
    subs = segment_for_format(words, cfg, fmt)
    out, report = validate_and_reflow(subs, cfg, fmt, "16-9")

    available = fmt.largeur_sequence_px * fmt.ratio_largeur_texte / (1 + cfg.marge_securite_rendu)
    m = FontMeasurer(font_path, fmt.taille_police)
    for sub in out:
        for line in sub.lines:
            assert m.width(line) <= available
    # Aucun mot perdu.
    assert sum(len(s.words) for s in out) == len(words)
    assert not report.warnings


def test_narrow_frame_forces_splits(font_path):
    """Un même texte peut tenir en 16:9 et nécessiter plus de sous-titres en
    9:16 (cadre plus étroit) — le mécanisme tourne indépendamment par format."""
    cfg = make_cfg(font_path)
    texts = ("une phrase suffisamment longue pour devoir être redistribuée "
             "sur plusieurs sous-titres en format vertical").split()
    words = make_words(texts, gap=0.05)

    fmt_wide = FormatConfig(taille_police=60, mots_max_par_ligne=10,
                            largeur_sequence_px=1920)
    fmt_narrow = FormatConfig(taille_police=60, mots_max_par_ligne=10,
                              largeur_sequence_px=1080)
    out_wide, _ = validate_and_reflow(
        segment_for_format(words, cfg, fmt_wide), cfg, fmt_wide, "16-9")
    out_narrow, rep = validate_and_reflow(
        segment_for_format(words, cfg, fmt_narrow), cfg, fmt_narrow, "9-16")

    assert len(out_narrow) >= len(out_wide)
    m = FontMeasurer(font_path, 60)
    available = 1080 * 0.75 / (1 + cfg.marge_securite_rendu)
    for sub in out_narrow:
        for line in sub.lines:
            assert m.width(line) <= available


def test_unbreakable_word_warning(font_path):
    """Garde-fou 4.7 : un élément insécable plus large que le cadre produit un
    avertissement explicite, pas une boucle infinie ni un échec silencieux."""
    cfg = make_cfg(font_path)
    fmt = FormatConfig(taille_police=200, largeur_sequence_px=400)
    sub = Subtitle(
        words=[Word("https://exemple.fr/tres/long/chemin/insecable", 0.0, 2.0)],
        start=0.0, end=2.0,
    )
    out, report = validate_and_reflow([sub], cfg, fmt, "9-16")
    assert len(out) == 1
    assert len(report.warnings) == 1
    assert "insécable" in report.warnings[0]


def test_split_preserves_word_timestamps(font_path):
    cfg = make_cfg(font_path)
    fmt = FormatConfig(taille_police=120, mots_max_par_ligne=10,
                       largeur_sequence_px=1080)
    words = make_words(
        "cette longue phrase déborde forcément du petit cadre vertical prévu".split(),
        gap=0.05)
    subs = segment_for_format(words, cfg, fmt)
    out, report = validate_and_reflow(subs, cfg, fmt, "9-16")
    # L'ordre et les timestamps des mots sont préservés à travers les scissions.
    flat = [w for s in out for w in s.words]
    assert [w.text for w in flat] == [w.text for w in words]
    for a, b in zip(out, out[1:]):
        assert a.start <= a.end
        assert a.end <= b.start + 1e-6
