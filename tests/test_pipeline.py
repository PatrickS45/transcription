"""Test d'intégration des étapes 2-5 (rejeu depuis JSON, sans GPU)."""

from cli import apply_args, build_parser
from config import PipelineConfig
from models import Segment, TranscriptionResult
from pipeline import run_pipeline
from tests.conftest import make_words


def make_transcription() -> TranscriptionResult:
    seg1 = Segment(words=make_words(
        "bonjour à tous et bienvenue dans cette nouvelle vidéo".split(), gap=0.05))
    seg2 = Segment(words=make_words(
        "aujourd'hui nous parlons de la validation par rendu réel".split(),
        start=seg1.end + 0.6, gap=0.05))
    return TranscriptionResult(segments=[seg1, seg2], engine="test", model="test")


def test_replay_generates_both_srt(tmp_path, font_path):
    """Critère 4 : les deux fichiers SRT sont générés en une seule exécution,
    sans retranscription redondante."""
    json_path = tmp_path / "video_transcription.json"
    make_transcription().save_json(json_path)

    cfg = PipelineConfig()
    cfg.police = font_path
    cfg.formats["16-9"].taille_police = 100
    cfg.formats["9-16"].taille_police = 80

    result = run_pipeline(
        str(tmp_path / "video.mp4"), cfg,
        out_dir=str(tmp_path), replay_json=str(json_path),
    )
    assert result.srt_files["16-9"].name == "video_16-9.srt"
    assert result.srt_files["9-16"].name == "video_9-16.srt"
    for path in result.srt_files.values():
        content = path.read_text(encoding="utf-8")
        assert "-->" in content
        assert "bonjour" in content.lower()
    assert "16-9" in result.summary() and "9-16" in result.summary()


def test_cli_args_override_defaults():
    parser = build_parser()
    args = parser.parse_args([
        "video.mp4", "--police", "font.ttf",
        "--taille-16-9", "110", "--taille-9-16", "64",
        "--seuil-silence", "300", "--marge", "0.05", "--device", "cpu",
        "--correction", "--correction-backend", "api",
    ])
    cfg = apply_args(PipelineConfig(), args)
    assert cfg.police == "font.ttf"
    assert cfg.formats["16-9"].taille_police == 110
    assert cfg.formats["9-16"].taille_police == 64
    assert cfg.seuil_silence_ms == 300
    assert cfg.marge_securite_rendu == 0.05
    assert cfg.device == "cpu"
    assert cfg.correction_llm.active is True
    assert cfg.correction_llm.backend == "api"


def test_missing_font_raises(tmp_path):
    json_path = tmp_path / "t.json"
    make_transcription().save_json(json_path)
    cfg = PipelineConfig()
    cfg.police = ""
    try:
        run_pipeline(str(tmp_path / "v.mp4"), cfg, replay_json=str(json_path))
        assert False, "une erreur explicite était attendue"
    except ValueError as e:
        assert "police" in str(e).lower()
