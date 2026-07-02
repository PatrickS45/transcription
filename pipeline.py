"""Orchestration du pipeline complet (étapes 1 → 5 du CDC §4).

Cœur pur Python indépendant de l'interface : utilisé à la fois par la CLI
(cli.py) et par l'interface locale (app.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from config import PipelineConfig
from correction import correct_transcription
from models import TranscriptionResult, merge_elision_fragments, sanitize_word_timing
from rendering import FontMeasurer, RenderReport, validate_and_reflow
from segmentation import segment_for_format
from srt_export import output_paths, write_srt


@dataclass
class PipelineResult:
    """Rapport final présenté à l'utilisateur (avertissements 4.7 inclus)."""

    transcription_json: Path | None = None
    srt_files: dict[str, Path] = field(default_factory=dict)
    reports: dict[str, RenderReport] = field(default_factory=dict)
    #: Avertissements du prétraitement (timestamps réparés, texte supprimé…).
    preprocess_warnings: list[str] = field(default_factory=list)

    @property
    def warnings(self) -> list[str]:
        return self.preprocess_warnings + [
            w for r in self.reports.values() for w in r.warnings
        ]

    def summary(self) -> str:
        lines = []
        if self.transcription_json:
            lines.append(f"Transcription intermédiaire : {self.transcription_json}")
        for name, path in self.srt_files.items():
            r = self.reports.get(name)
            detail = ""
            if r:
                detail = (
                    f" — {r.subtitles_out} sous-titres "
                    f"(cadre {r.available_width_px:.0f}px utiles, "
                    f"taille {r.font_size:g}, {r.splits} scission(s))"
                )
            lines.append(f"SRT {name} : {path}{detail}")
        if self.warnings:
            lines.append("")
            lines.append(f"⚠ {len(self.warnings)} avertissement(s) :")
            lines.extend(f"  - {w}" for w in self.warnings)
        else:
            lines.append("Aucun avertissement : toutes les lignes tiennent dans le cadre.")
        return "\n".join(lines)


def transcribe_step(
    video_path: str,
    cfg: PipelineConfig,
    progress: Callable[[str], None] | None = None,
) -> TranscriptionResult:
    """Étape 1 — transcription via le backend sélectionné."""
    from transcription import get_backend

    backend = get_backend(cfg.moteur_transcription)
    return backend.transcribe(
        video_path,
        model=cfg.modele,
        language=cfg.langue,
        device=cfg.device,
        seuil_silence_ms=cfg.seuil_silence_ms,
        progress=progress,
    )


def layout_and_export(
    result: TranscriptionResult,
    cfg: PipelineConfig,
    video_path: str,
    out_dir: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> PipelineResult:
    """Étapes 2 → 5 pour les deux formats, à partir d'une transcription
    (fraîche ou rejouée depuis le JSON — pas de retranscription)."""
    notify = progress or (lambda msg: None)
    if not cfg.police:
        raise ValueError("Le chemin du fichier de police (.ttf/.otf) est requis.")
    if not Path(cfg.police).is_file():
        raise FileNotFoundError(f"Fichier de police introuvable : {cfg.police}")

    out = PipelineResult()
    paths = output_paths(video_path, out_dir)
    words = result.words
    if not words:
        raise ValueError("Transcription vide : aucun mot horodaté à mettre en page.")

    for name, fmt in cfg.formats.items():
        notify(f"[{name}] Segmentation sur les pauses (seuil {cfg.seuil_silence_ms} ms)…")
        subtitles = segment_for_format(words, cfg, fmt)

        notify(
            f"[{name}] Validation par rendu réel (police {Path(cfg.police).name}, "
            f"taille {fmt.taille_police:g})…"
        )
        measurer = FontMeasurer(cfg.police, fmt.taille_police)
        subtitles, report = validate_and_reflow(subtitles, cfg, fmt, name, measurer)
        out.reports[name] = report

        path = paths[name]
        write_srt(subtitles, path, bom=cfg.srt_bom)
        out.srt_files[name] = path
        notify(f"[{name}] Export : {path}")
    return out


def run_pipeline(
    video_path: str,
    cfg: PipelineConfig,
    out_dir: str | None = None,
    replay_json: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> PipelineResult:
    """Pipeline complet. Si `replay_json` est fourni, rejoue les étapes 2 à 5
    depuis la transcription sauvegardée, sans retranscrire (itération rapide
    sur les paramètres de mise en page)."""
    notify = progress or (lambda msg: None)

    if replay_json:
        notify(f"Rejeu depuis la transcription sauvegardée : {replay_json}")
        result = TranscriptionResult.load_json(replay_json)
        json_path = Path(replay_json)
        # Corrige rétroactivement les fragments élidés/composés d'anciennes
        # transcriptions sauvegardées avant ce correctif (pas de retranscription).
        result = merge_elision_fragments(result)
        preprocess_warnings = sanitize_word_timing(result, cfg.car_sec_max)
    else:
        notify(
            f"Transcription ({cfg.moteur_transcription}, modèle {cfg.modele}, "
            f"device {cfg.device}, langue forcée {cfg.langue})…"
        )
        result = transcribe_step(video_path, cfg, progress)
        result = merge_elision_fragments(result)
        preprocess_warnings = sanitize_word_timing(result, cfg.car_sec_max)
        base = Path(out_dir) if out_dir else Path(video_path).parent
        json_path = base / f"{Path(video_path).stem}_transcription.json"
        result.save_json(json_path)
        notify(f"Transcription intermédiaire sauvegardée : {json_path}")

        if cfg.correction_llm.active:
            notify(f"Correction LLM ({cfg.correction_llm.backend})…")
            result = correct_transcription(result, cfg.correction_llm, progress)
            corrected_path = json_path.with_name(json_path.stem + "_corrigee.json")
            result.save_json(corrected_path)
            json_path = corrected_path
            notify(f"Transcription corrigée sauvegardée : {corrected_path}")

    out = layout_and_export(result, cfg, video_path, out_dir, progress)
    out.transcription_json = json_path
    out.preprocess_warnings = preprocess_warnings
    for w in preprocess_warnings:
        notify(f"⚠ {w}")
    return out
