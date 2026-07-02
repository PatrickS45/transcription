"""Contrat commun des moteurs de transcription et registre des backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from models import TranscriptionResult


class TranscriptionBackend(ABC):
    """Interface commune : toute implémentation restitue le format
    intermédiaire (segments avec texte + timestamps, au niveau du mot).

    La détection de pauses fait partie du contrat : si le moteur ne fournit
    pas de VAD, l'implémentation doit appeler un VAD séparé (Silero) pour que
    les silences soient exploitables via les gaps entre timestamps de mots.
    """

    name: str = ""

    @abstractmethod
    def transcribe(
        self,
        audio_path: str,
        model: str,
        language: str = "fr",
        device: str = "cuda",
        seuil_silence_ms: int = 250,
        progress: Callable[[str], None] | None = None,
    ) -> TranscriptionResult:
        """Transcrit `audio_path` et retourne le format intermédiaire commun.

        `language` est toujours forcée (jamais d'auto-détection).
        """


_REGISTRY: dict[str, Callable[[], TranscriptionBackend]] = {}


def register_backend(name: str, factory: Callable[[], TranscriptionBackend]) -> None:
    _REGISTRY[name] = factory


def list_backends() -> list[str]:
    return sorted(_REGISTRY)


def get_backend(name: str) -> TranscriptionBackend:
    if name not in _REGISTRY:
        raise ValueError(
            f"Moteur de transcription inconnu : {name!r}. "
            f"Disponibles : {', '.join(list_backends())}"
        )
    return _REGISTRY[name]()


def _faster_whisper_factory() -> TranscriptionBackend:
    from transcription.faster_whisper_backend import FasterWhisperBackend

    return FasterWhisperBackend()


def _whisperx_factory() -> TranscriptionBackend:
    from transcription.whisperx_backend import WhisperXBackend

    return WhisperXBackend()


register_backend("faster-whisper", _faster_whisper_factory)
register_backend("whisperx", _whisperx_factory)
