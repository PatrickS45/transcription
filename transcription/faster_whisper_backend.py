"""Moteur par défaut : faster-whisper (local, VAD Silero intégré)."""

from __future__ import annotations

from typing import Callable

from models import Segment, TranscriptionResult, Word
from transcription.base import TranscriptionBackend


class FasterWhisperBackend(TranscriptionBackend):
    name = "faster-whisper"

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
            from faster_whisper import WhisperModel
        except ImportError as e:  # pragma: no cover - dépend de l'environnement
            raise RuntimeError(
                "faster-whisper n'est pas installé : pip install faster-whisper"
            ) from e

        notify = progress or (lambda msg: None)
        notify(
            f"Chargement du modèle {model} sur {device}… (premier lancement : "
            "téléchargement depuis Hugging Face, plusieurs minutes selon la "
            "connexion — les lancements suivants seront quasi instantanés, "
            "modèle mis en cache)"
        )
        compute_type = "float16" if device == "cuda" else "int8"
        whisper = WhisperModel(model, device=device, compute_type=compute_type)

        notify("Transcription en cours (langue forcée : fr, VAD activé)…")
        segments_iter, info = whisper.transcribe(
            audio_path,
            language=language,  # forcée, jamais d'auto-détection
            word_timestamps=True,  # timestamps au niveau du mot (contrat §4)
            vad_filter=True,  # VAD Silero intégré (étape 2)
            vad_parameters={"min_silence_duration_ms": int(seuil_silence_ms)},
        )

        result = TranscriptionResult(
            language=language,
            engine=self.name,
            model=model,
            audio_duration=float(getattr(info, "duration", 0.0)),
        )
        for seg in segments_iter:
            words = [
                Word(text=w.word.strip(), start=float(w.start), end=float(w.end))
                for w in (seg.words or [])
                if w.word.strip()
            ]
            if words:
                result.segments.append(Segment(words=words))
            notify(f"Segment transcrit jusqu'à {seg.end:.1f}s")
        return result
