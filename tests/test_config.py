from config import PipelineConfig


def test_defaults_match_cdc():
    cfg = PipelineConfig()
    assert cfg.langue == "fr"
    assert cfg.moteur_transcription == "faster-whisper"
    assert cfg.modele == "large-v3"
    assert cfg.correction_llm.active is False
    assert cfg.duree_min_ms == 833
    assert cfg.duree_max_ms == 7000
    assert cfg.ecart_min_ms == 83
    assert cfg.car_sec_max == 20.0
    assert cfg.seuil_silence_ms == 250
    assert cfg.marge_securite_rendu == 0.04
    assert cfg.formats["16-9"].mots_max_par_ligne == 7
    assert cfg.formats["9-16"].mots_max_par_ligne == 4
    assert cfg.formats["16-9"].ratio_largeur_texte == 0.75
    assert cfg.formats["9-16"].ratio_largeur_texte == 0.75
    assert cfg.formats["16-9"].lignes_max == 2


def test_preset_roundtrip(tmp_path):
    cfg = PipelineConfig()
    cfg.police = "Atkinson.ttf"
    cfg.formats["16-9"].taille_police = 100
    cfg.formats["9-16"].taille_police = 72
    cfg.seuil_silence_ms = 300
    path = tmp_path / "projet.json"
    cfg.save_preset(path)

    loaded = PipelineConfig.load_preset(path)
    assert loaded.police == "Atkinson.ttf"
    assert loaded.formats["16-9"].taille_police == 100
    assert loaded.formats["9-16"].taille_police == 72
    assert loaded.seuil_silence_ms == 300


def test_api_key_never_saved(tmp_path):
    cfg = PipelineConfig()
    cfg.correction_llm.api_key = "sk-secret"
    path = tmp_path / "projet.json"
    cfg.save_preset(path)
    assert "sk-secret" not in path.read_text(encoding="utf-8")
