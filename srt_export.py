"""Étape 5 — export SRT standard, UTF-8, deux fichiers par exécution (CDC §4)."""

from __future__ import annotations

from pathlib import Path

from models import Subtitle


def format_timestamp(seconds: float) -> str:
    """Timestamp SRT `HH:MM:SS,mmm` (virgule, format standard)."""
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    m = (total_s // 60) % 60
    h = total_s // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def render_srt(subtitles: list[Subtitle]) -> str:
    """Contenu SRT : numérotation, timestamps, texte, ligne vide de séparation."""
    blocks = []
    for i, sub in enumerate(subtitles, start=1):
        text = "\n".join(sub.lines) if sub.lines else sub.text
        blocks.append(
            f"{i}\n{format_timestamp(sub.start)} --> {format_timestamp(sub.end)}\n{text}\n"
        )
    return "\n".join(blocks)


def write_srt(subtitles: list[Subtitle], path: str | Path, bom: bool = False) -> Path:
    """Écrit le fichier SRT en UTF-8 (obligatoire pour les accents français).

    La présence du BOM est un paramètre, à valider contre le comportement
    d'import de Premiere lors des premiers tests.
    """
    path = Path(path)
    encoding = "utf-8-sig" if bom else "utf-8"
    path.write_text(render_srt(subtitles), encoding=encoding)
    return path


def output_paths(video_path: str | Path, out_dir: str | Path | None = None) -> dict[str, Path]:
    """Chemins de sortie : `{nom_video}_16-9.srt` et `{nom_video}_9-16.srt`."""
    video = Path(video_path)
    base = Path(out_dir) if out_dir else video.parent
    stem = video.stem
    return {
        "16-9": base / f"{stem}_16-9.srt",
        "9-16": base / f"{stem}_9-16.srt",
    }
