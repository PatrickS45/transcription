"""Format intermédiaire commun du pipeline (contrat de l'étape 1 du CDC §4).

Toute implémentation de moteur de transcription doit produire une
`TranscriptionResult` : liste de segments avec texte + timestamps début/fin,
avec timestamps au niveau du mot. Ce format est sauvegardé en JSON et sert de
source unique aux deux formats de sortie (rejeu sans retranscription).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

INTERMEDIATE_FORMAT_VERSION = 1


@dataclass
class Word:
    """Un mot avec ses timestamps (en secondes)."""

    text: str
    start: float
    end: float

    def to_dict(self) -> dict:
        return {"text": self.text, "start": self.start, "end": self.end}

    @classmethod
    def from_dict(cls, d: dict) -> "Word":
        return cls(text=d["text"], start=float(d["start"]), end=float(d["end"]))


@dataclass
class Segment:
    """Un segment de parole continue (entre deux silences détectés par le VAD)."""

    words: list[Word] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def start(self) -> float:
        return self.words[0].start if self.words else 0.0

    @property
    def end(self) -> float:
        return self.words[-1].end if self.words else 0.0

    def to_dict(self) -> dict:
        return {"words": [w.to_dict() for w in self.words]}

    @classmethod
    def from_dict(cls, d: dict) -> "Segment":
        return cls(words=[Word.from_dict(w) for w in d["words"]])


@dataclass
class TranscriptionResult:
    """Sortie intermédiaire de l'étape 1, sérialisable en JSON."""

    segments: list[Segment] = field(default_factory=list)
    language: str = "fr"
    engine: str = ""
    model: str = ""
    audio_duration: float = 0.0

    @property
    def words(self) -> list[Word]:
        return [w for seg in self.segments for w in seg.words]

    @property
    def text(self) -> str:
        return " ".join(seg.text for seg in self.segments)

    def to_dict(self) -> dict:
        return {
            "format_version": INTERMEDIATE_FORMAT_VERSION,
            "language": self.language,
            "engine": self.engine,
            "model": self.model,
            "audio_duration": self.audio_duration,
            "segments": [s.to_dict() for s in self.segments],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TranscriptionResult":
        return cls(
            segments=[Segment.from_dict(s) for s in d["segments"]],
            language=d.get("language", "fr"),
            engine=d.get("engine", ""),
            model=d.get("model", ""),
            audio_duration=float(d.get("audio_duration", 0.0)),
        )

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "TranscriptionResult":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class Subtitle:
    """Un sous-titre affichable : mots + timestamps, puis lignes après mise en page.

    Les lignes ne sont définitives qu'après la validation par rendu réel
    (étape 4) ; avant cela, `lines` est une mise en page indicative.
    """

    words: list[Word] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)
    start: float = 0.0
    end: float = 0.0

    @property
    def text(self) -> str:
        return "\n".join(self.lines) if self.lines else " ".join(w.text for w in self.words)

    @property
    def duration_ms(self) -> float:
        return (self.end - self.start) * 1000.0

    @property
    def char_count(self) -> int:
        return sum(len(w.text) for w in self.words) + max(0, len(self.words) - 1)

    @property
    def chars_per_second(self) -> float:
        dur = self.end - self.start
        return self.char_count / dur if dur > 0 else float("inf")
