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


def merge_elision_fragments(result: TranscriptionResult) -> TranscriptionResult:
    """Recolle les fragments de mots que certains moteurs tokenisent
    séparément de part et d'autre d'une apostrophe ou d'un trait d'union
    (ex. "d" + "'associé" -> "d'associé", "c'est" + "-à" + "-dire"), pour
    éviter l'espace parasite qui apparaît sinon dans les sous-titres.

    Appliquée sur le résultat brut du moteur de transcription, avant toute
    autre étape, de façon à ce que segmentation et correction LLM travaillent
    sur des mots déjà correctement recollés.
    """
    for seg in result.segments:
        merged: list[Word] = []
        for w in seg.words:
            if merged and w.text and w.text[0] in ("'", "-", "’"):
                prev = merged[-1]
                merged[-1] = Word(text=prev.text + w.text, start=prev.start, end=w.end)
            else:
                merged.append(w)
        seg.words = merged
    return result


def sanitize_word_timing(
    result: TranscriptionResult,
    car_sec_max: float = 20.0,
    max_expand: int = 12,
) -> list[str]:
    """Répare les timestamps de mots physiquement impossibles.

    Les moteurs de transcription produisent parfois des mots (voire des
    phrases entières) compressés sur quelques millisecondes — typiquement
    autour d'une hallucination ou d'une frontière de chunk VAD. Ces
    timestamps corrompus produisent des sous-titres flash illisibles que la
    mise en page ne peut pas rattraper.

    Stratégie : chaque suite de mots à vitesse impossible (>50 car/s par mot)
    est réétalée proportionnellement aux caractères sur une fenêtre élargie
    aux mots voisins (sans franchir un silence > 1 s), jusqu'à retrouver une
    vitesse de parole plausible. Si aucune fenêtre plausible n'existe, le
    texte est supprimé avec un avertissement explicite (probable
    hallucination : le temps nécessaire pour le prononcer n'existe pas dans
    l'audio). Retourne la liste des avertissements.
    """
    def corrupted(w: Word) -> bool:
        d = w.end - w.start
        return d <= 0 or len(w.text) / d > 50.0

    warnings: list[str] = []
    flat = [
        (si, wi)
        for si, seg in enumerate(result.segments)
        for wi in range(len(seg.words))
    ]

    def get(k: int) -> Word:
        si, wi = flat[k]
        return result.segments[si].words[wi]

    n = len(flat)
    to_drop: set[tuple[int, int]] = set()
    tolerance = car_sec_max * 1.25
    i = 0
    while i < n:
        if not corrupted(get(i)):
            i += 1
            continue
        j = i
        while j < n and corrupted(get(j)):
            j += 1
        lo, hi = i, j

        def window_cps() -> float:
            chars = sum(len(get(k).text) + 1 for k in range(lo, hi)) - 1
            dur = get(hi - 1).end - get(lo).start
            return chars / dur if dur > 0 else float("inf")

        for _ in range(max_expand):
            if window_cps() <= tolerance:
                break
            if hi < n and (get(hi).start - get(hi - 1).end) < 1.0:
                hi += 1
            elif lo > 0 and (get(lo).start - get(lo - 1).end) < 1.0:
                lo -= 1
            else:
                break

        run_text = " ".join(get(k).text for k in range(i, j))
        # Un ou deux mots isolés ne sont jamais supprimés (trop de risque de
        # perdre du texte réel) : réétalement au mieux sur la fenêtre, la
        # fusion de sous-titres en aval absorbe la densité résiduelle. Seules
        # les suites de 3 mots et plus (échelle d'une phrase hallucinée)
        # peuvent être supprimées.
        if window_cps() <= tolerance or (j - i) <= 2:
            t0 = get(lo).start
            span = get(hi - 1).end - t0
            total = sum(len(get(k).text) + 1 for k in range(lo, hi)) - 1
            pos = 0
            for k in range(lo, hi):
                w = get(k)
                w.start = t0 + span * pos / total
                pos += len(w.text)
                w.end = t0 + span * pos / total
                pos += 1
            if j - i >= 5:
                warnings.append(
                    f"Timestamps incohérents recalés vers "
                    f"{int(t0 // 60):02d}:{int(t0 % 60):02d} sur « {run_text} » "
                    "— passage possiblement halluciné, à vérifier à l'écoute."
                )
        else:
            t0 = get(i).start
            warnings.append(
                f"Texte supprimé vers {int(t0 // 60):02d}:{int(t0 % 60):02d} : "
                f"« {run_text} » — vitesse de parole impossible, le temps pour "
                "le prononcer n'existe pas dans l'audio (probable hallucination "
                "du moteur). Vérifier ce passage à l'écoute."
            )
            for k in range(i, j):
                to_drop.add(flat[k])
        i = max(hi, j)

    if to_drop:
        for si, seg in enumerate(result.segments):
            seg.words = [w for wi, w in enumerate(seg.words) if (si, wi) not in to_drop]
        result.segments = [s for s in result.segments if s.words]
    return warnings


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
