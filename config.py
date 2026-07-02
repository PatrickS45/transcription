"""Paramètres du pipeline, valeurs par défaut et préréglages par projet (CDC §5).

Toutes les valeurs ci-dessous sont des défauts de départ, réglables depuis
l'interface ou la ligne de commande à chaque exécution. Aucune valeur n'est
figée : en particulier `taille_police` n'a pas de défaut sensé universel et
doit être fournie à chaque exécution, indépendamment pour le 16:9 et le 9:16.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class FormatConfig:
    """Contraintes de mise en page propres à un format de sortie (16:9 ou 9:16)."""

    #: Taille de police du projet en cours — variable selon le projet,
    #: fournie à chaque exécution (jamais une constante).
    taille_police: float = 100.0
    #: Largeur de la zone de texte en ratio de la largeur de séquence (CDC §6).
    ratio_largeur_texte: float = 0.75
    mots_max_par_ligne: int = 7
    lignes_max: int = 2
    #: Indicatif — le rendu réel fait foi. 0 = désactivé.
    caracteres_max_par_ligne: int = 0
    #: Largeur de la séquence de référence en pixels (1920 en 16:9, 1080 en 9:16).
    largeur_sequence_px: int = 1920


@dataclass
class CorrectionConfig:
    """Étape 1bis — correction du texte par LLM (optionnelle, désactivée par défaut)."""

    active: bool = False
    backend: str = "ollama"  # "ollama" (local) | "api" (ex. Claude)
    modele_llm: str = ""
    #: Chemin vers un fichier texte de noms propres / termes métier (optionnel).
    glossaire: str = ""
    #: URL du serveur Ollama local.
    ollama_url: str = "http://localhost:11434"
    #: Clé API pour le backend cloud (jamais sauvegardée dans les préréglages).
    api_key: str = ""


@dataclass
class PipelineConfig:
    """Ensemble des paramètres exposés (CDC §5)."""

    langue: str = "fr"  # fixe : jamais d'auto-détection
    moteur_transcription: str = "faster-whisper"
    modele: str = "large-v3"
    device: str = "cuda"  # repli "cpu" possible (lent, tests uniquement)
    correction_llm: CorrectionConfig = field(default_factory=CorrectionConfig)
    #: Chemin vers le fichier .ttf/.otf (Atkinson Hyperlegible).
    police: str = ""
    formats: dict[str, FormatConfig] = field(
        default_factory=lambda: {
            "16-9": FormatConfig(mots_max_par_ligne=7, largeur_sequence_px=1920),
            "9-16": FormatConfig(mots_max_par_ligne=4, largeur_sequence_px=1080),
        }
    )
    duree_min_ms: int = 833  # référence Netflix
    duree_max_ms: int = 7000
    ecart_min_ms: int = 83
    car_sec_max: float = 20.0
    seuil_silence_ms: int = 250
    #: Marge de sécurité appliquée à la largeur mesurée (écarts Pillow/Premiere, §4.8).
    marge_securite_rendu: float = 0.04
    #: Présence du BOM UTF-8 dans les SRT (à valider contre l'import Premiere, §4 étape 5).
    srt_bom: bool = False

    # ------------------------------------------------------------------ presets

    def to_dict(self) -> dict:
        d = asdict(self)
        # La clé API ne doit jamais être écrite sur disque.
        d["correction_llm"]["api_key"] = ""
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        d = dict(d)
        correction = CorrectionConfig(**d.pop("correction_llm", {}))
        formats = {
            name: FormatConfig(**fmt) for name, fmt in d.pop("formats", {}).items()
        }
        cfg = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        cfg.correction_llm = correction
        if formats:
            cfg.formats = formats
        return cfg

    def save_preset(self, path: str | Path) -> None:
        """Sauvegarde des réglages du projet (préréglage rechargeable, CDC §11)."""
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load_preset(cls, path: str | Path) -> "PipelineConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
