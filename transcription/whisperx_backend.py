"""Moteur alternatif : WhisperX (alignement forcé des timestamps mot par mot).

Prévu par l'architecture (CDC §4 étape 1) — implémentation best-effort,
activable dès que `whisperx` est installé.
"""

from __future__ import annotations

from typing import Callable

from models import Segment, TranscriptionResult, Word
from transcription.base import TranscriptionBackend


class WhisperXBackend(TranscriptionBackend):
    name = "whisperx"

    def transcribe(
        self,
        audio_path: str,
        model: str,
        language: str = "fr",
        device: str = "cuda",
        seuil_silence_ms: int = 250,
        progress: Callable[[str], None] | None = None,
    ) -> TranscriptionResult:
        try:
            import whisperx
        except ImportError as e:  # pragma: no cover - dépend de l'environnement
            raise RuntimeError(
                "whisperx n'est pas installé : pip install whisperx"
            ) from e

        notify = progress or (lambda msg: None)
        compute_type = "float16" if device == "cuda" else "int8"

        notify(f"Chargement du modèle {model} sur {device}…")
        asr = whisperx.load_model(model, device, compute_type=compute_type, language=language)
        audio = whisperx.load_audio(audio_path)

        notify("Transcription en cours (langue forcée : fr)…")
        raw = asr.transcribe(audio, language=language)

        notify("Alignement forcé des timestamps mot par mot…")
        align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
        aligned = whisperx.align(
            raw["segments"], align_model, metadata, audio, device, return_char_alignments=False
        )

        result = TranscriptionResult(language=language, engine=self.name, model=model)
        for seg in aligned["segments"]:
            words = []
            for w in seg.get("words", []):
                text = str(w.get("word", "")).strip()
                if not text or "start" not in w or "end" not in w:
                    continue
                words.append(Word(text=text, start=float(w["start"]), end=float(w["end"])))
            if words:
                result.segments.append(Segment(words=words))
        if result.segments:
            result.audio_duration = result.segments[-1].end
        return result
