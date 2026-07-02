"""Étapes 2-3 — découpage en sous-titres sur les pauses + contraintes dures.

Pur Python, testable sans GPU. Le découpage s'aligne sur les silences
détectés (gaps entre mots >= seuil_silence_ms) ; en cas de conflit entre
pause naturelle et contraintes dures, les contraintes dures priment (CDC §4
étape 3).
"""

from __future__ import annotations

from dataclasses import dataclass

from config import FormatConfig, PipelineConfig
from models import Subtitle, Word


@dataclass
class HardConstraints:
    """Contraintes dures applicables à chaque sous-titre, par format."""

    max_words: int  # mots_max_par_ligne * lignes_max
    max_chars: int  # caracteres_max_par_ligne * lignes_max (0 = désactivé)
    duree_min_ms: float
    duree_max_ms: float
    ecart_min_ms: float
    car_sec_max: float

    @classmethod
    def from_config(cls, cfg: PipelineConfig, fmt: FormatConfig) -> "HardConstraints":
        return cls(
            max_words=fmt.mots_max_par_ligne * fmt.lignes_max,
            max_chars=fmt.caracteres_max_par_ligne * fmt.lignes_max,
            duree_min_ms=cfg.duree_min_ms,
            duree_max_ms=cfg.duree_max_ms,
            ecart_min_ms=cfg.ecart_min_ms,
            car_sec_max=cfg.car_sec_max,
        )


def _char_count(words: list[Word]) -> int:
    return sum(len(w.text) for w in words) + max(0, len(words) - 1)


def _block_violates(words: list[Word], c: HardConstraints) -> bool:
    """Vrai si le bloc dépasse une contrainte dure (hors durée min / cps,
    traitées après coup par extension des timestamps)."""
    if len(words) > c.max_words:
        return True
    if c.max_chars and _char_count(words) > c.max_chars:
        return True
    if (words[-1].end - words[0].start) * 1000.0 > c.duree_max_ms:
        return True
    return False


def segment_words(
    words: list[Word],
    constraints: HardConstraints,
    seuil_silence_ms: float,
) -> list[Subtitle]:
    """Découpe une liste de mots horodatés en sous-titres.

    Stratégie : accumulation gloutonne ; on coupe systématiquement sur les
    silences >= seuil, et quand une contrainte dure force une coupure au
    milieu d'un flux continu, on recule si possible jusqu'au dernier point
    de pause du bloc courant (jamais couper au milieu d'un groupe de mots
    séparé par un silence court).
    """
    if not words:
        return []

    subtitles: list[Subtitle] = []
    block: list[Word] = []
    # Index (dans block) du début du dernier groupe après pause, pour backtrack.
    last_pause_idx = 0

    def flush(upto: int | None = None) -> None:
        nonlocal block, last_pause_idx
        emit = block if upto is None else block[:upto]
        rest = [] if upto is None else block[upto:]
        if emit:
            subtitles.append(
                Subtitle(words=emit, start=emit[0].start, end=emit[-1].end)
            )
        block = rest
        last_pause_idx = 0

    for i, word in enumerate(words):
        if block:
            gap_ms = (word.start - block[-1].end) * 1000.0
            if gap_ms >= seuil_silence_ms:
                # Pause naturelle : point de coupure privilégié.
                if (block[-1].end - block[0].start) * 1000.0 >= constraints.duree_min_ms:
                    flush()
                else:
                    # Bloc trop court pour être émis seul : on note la pause
                    # comme point de coupure candidat et on continue.
                    last_pause_idx = len(block)
        candidate = block + [word]
        if block and _block_violates(candidate, constraints):
            # Contrainte dure atteinte : couper de préférence sur la
            # dernière pause du bloc, sinon juste avant ce mot.
            if 0 < last_pause_idx < len(block):
                flush(upto=last_pause_idx)
            else:
                flush()
        block.append(word)
    flush()

    subtitles = _merge_timing_anomalies(subtitles, constraints)
    _enforce_timing(subtitles, constraints)
    return subtitles


def _merge_timing_anomalies(
    subtitles: list[Subtitle], c: HardConstraints
) -> list[Subtitle]:
    """Fusionne les sous-titres dont la vitesse de lecture instantanée est
    physiquement impossible (indice de timestamps corrompus, ex. duplication/
    hallucination du moteur de transcription à la frontière d'un chunk VAD)
    avec le sous-titre voisin, plutôt que de produire un flash illisible de
    quelques millisecondes qui violerait silencieusement la durée minimale.
    """
    threshold = c.car_sec_max * 3
    result: list[Subtitle] = []
    i = 0
    while i < len(subtitles):
        sub = subtitles[i]
        if sub.chars_per_second > threshold and i + 1 < len(subtitles):
            nxt = subtitles[i + 1]
            words = sub.words + nxt.words
            subtitles[i + 1] = Subtitle(words=words, start=words[0].start, end=words[-1].end)
            i += 1
            continue
        if sub.chars_per_second > threshold and result:
            prev = result.pop()
            words = prev.words + sub.words
            result.append(Subtitle(words=words, start=words[0].start, end=words[-1].end))
            i += 1
            continue
        result.append(sub)
        i += 1
    return result


def _enforce_timing(subtitles: list[Subtitle], c: HardConstraints) -> None:
    """Applique durée min, vitesse de lecture max et écart min entre sous-titres.

    On ne peut pas raccourcir le texte ici : on étend la durée d'affichage
    dans la limite du sous-titre suivant (moins l'écart min) et de la durée
    max.
    """
    gap_s = c.ecart_min_ms / 1000.0
    for i, sub in enumerate(subtitles):
        needed_s = max(
            c.duree_min_ms / 1000.0,
            sub.char_count / c.car_sec_max if c.car_sec_max > 0 else 0.0,
        )
        needed_s = min(needed_s, c.duree_max_ms / 1000.0)
        target_end = sub.start + needed_s
        limit = subtitles[i + 1].start - gap_s if i + 1 < len(subtitles) else float("inf")
        sub.end = min(max(sub.end, target_end), max(limit, sub.end))
        # Écart min avec le suivant : on rogne la fin si nécessaire.
        if i + 1 < len(subtitles) and sub.end > subtitles[i + 1].start - gap_s:
            sub.end = max(subtitles[i + 1].start - gap_s, sub.start + 0.001)


def layout_lines(
    words: list[str],
    max_lines: int,
    max_words_per_line: int,
    max_chars_per_line: int = 0,
) -> list[str] | None:
    """Répartit des mots sur au plus `max_lines` lignes équilibrées.

    Retourne None si impossible en respectant mots/ligne et caractères/ligne.
    Mise en page indicative : la mise en page définitive est décidée par le
    rendu réel (rendering.py).
    """
    if not words:
        return []

    def fits(line: list[str]) -> bool:
        if len(line) > max_words_per_line:
            return False
        if max_chars_per_line and len(" ".join(line)) > max_chars_per_line:
            return False
        return True

    # Recherche du découpage en k lignes (k minimal) minimisant l'écart de
    # longueur entre lignes (lignes équilibrées, plus agréables à lire).
    for k in range(1, max_lines + 1):
        best: list[list[str]] | None = None
        best_score = float("inf")

        def rec(start: int, lines_left: int, acc: list[list[str]]) -> None:
            nonlocal best, best_score
            if lines_left == 1:
                line = words[start:]
                if not fits(line):
                    return
                lines = acc + [line]
                lengths = [len(" ".join(l)) for l in lines]
                score = max(lengths) - min(lengths)
                if score < best_score:
                    best, best_score = lines, score
                return
            for cut in range(start + 1, len(words) - lines_left + 2):
                line = words[start:cut]
                if not fits(line):
                    break
                rec(cut, lines_left - 1, acc + [line])

        if k <= len(words):
            rec(0, k, [])
        if best is not None:
            return [" ".join(line) for line in best]
    return None


def segment_for_format(
    words: list[Word],
    cfg: PipelineConfig,
    fmt: FormatConfig,
) -> list[Subtitle]:
    """Segmentation initiale complète pour un format (étapes 2-3)."""
    constraints = HardConstraints.from_config(cfg, fmt)
    subtitles = segment_words(words, constraints, cfg.seuil_silence_ms)
    for sub in subtitles:
        lines = layout_lines(
            [w.text for w in sub.words],
            fmt.lignes_max,
            fmt.mots_max_par_ligne,
            fmt.caracteres_max_par_ligne,
        )
        sub.lines = lines if lines is not None else [" ".join(w.text for w in sub.words)]
    return subtitles
