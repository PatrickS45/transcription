"""Backends de transcription interchangeables (CDC §4 étape 1).

Chaque backend implémente le contrat commun `TranscriptionBackend` et
restitue une `TranscriptionResult` (segments + mots + timestamps). Tout
futur moteur (ASR local ou API cloud) s'ajoute ici sans refonte.
"""

from transcription.base import TranscriptionBackend, get_backend, list_backends

__all__ = ["TranscriptionBackend", "get_backend", "list_backends"]
