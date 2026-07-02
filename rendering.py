"""Étape 4 — validation par rendu réel (pièce maîtresse du CDC).

Mesure la largeur réelle en pixels de chaque ligne avec la police du projet
(Pillow/FreeType) à la taille fournie pour cette exécution, la compare à la
largeur de cadre disponible du format, et reformate automatiquement jusqu'à
convergence. Pur Python, testable sans GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from PIL import ImageFont

from config import FormatConfig, PipelineConfig
from models import Subtitle, Word
from segmentation import HardConstraints, _enforce_timing

MAX_REFLOW_ITERATIONS = 50


class FontMeasurer:
    """Mesure de largeur de texte rendu, pour une police et une taille données.

    La taille de police est un paramètre d'entrée fourni à chaque exécution —
    jamais une valeur par défaut supposée (CDC §4 étape 4.2-4.3).
    """

    def __init__(self, font_path: str, font_size: float):
        self.font_path = font_path
        self.font_size = font_size
        self._font = ImageFont.truetype(font_path, int(round(font_size)))
        self._cache: dict[str, float] = {}

    def width(self, text: str) -> float:
        """Largeur réelle en pixels du texte à la taille de police fournie."""
        if text not in self._cache:
            self._cache[text] = float(self._font.getlength(text))
        return self._cache[text]


@dataclass
class RenderReport:
    """Rapport de sortie de la validation (avertissements de l'étape 4.7 inclus)."""

    format_name: str = ""
    frame_width_px: float = 0.0
    available_width_px: float = 0.0
    font_size: float = 0.0
    subtitles_in: int = 0
    subtitles_out: int = 0
    reflowed: int = 0
    splits: int = 0
    warnings: list[str] = field(default_factory=list)


def _layout_by_width(
    word_texts: list[str],
    measurer: FontMeasurer,
    available_px: float,
    max_lines: int,
    max_words_per_line: int,
    max_chars_per_line: int,
) -> list[str] | None:
    """Mise en page pixel-réelle : répartit les mots sur au plus `max_lines`
    lignes tenant chacune dans `available_px`, en équilibrant les largeurs.

    Retourne None si aucune répartition ne tient (il faut alors scinder le
    sous-titre).
    """
    if not word_texts:
        return []

    def fits(line: list[str]) -> bool:
        if len(line) > max_words_per_line:
            return False
        text = " ".join(line)
        if max_chars_per_line and len(text) > max_chars_per_line:
            return False
        return measurer.width(text) <= available_px

    for k in range(1, max_lines + 1):
        best: list[str] | None = None
        best_score = float("inf")

        def rec(start: int, lines_left: int, acc: list[str]) -> None:
            nonlocal best, best_score
            if lines_left == 1:
                line = word_texts[start:]
                if not fits(line):
                    return
                lines = acc + [" ".join(line)]
                widths = [measurer.width(l) for l in lines]
                score = max(widths) - min(widths)
                if score < best_score:
                    best, best_score = lines, score
                return
            for cut in range(start + 1, len(word_texts) - lines_left + 2):
                line = word_texts[start:cut]
                if not fits(line):
                    break
                rec(cut, lines_left - 1, acc + [" ".join(line)])

        if k <= len(word_texts):
            rec(0, k, [])
        if best is not None:
            return best
    return None


def _split_subtitle(sub: Subtitle, duree_min_ms: float = 0.0) -> tuple[Subtitle, Subtitle]:
    """Scinde un sous-titre en deux à la meilleure frontière de mot.

    Le point de coupure privilégie, dans l'ordre : la ponctuation de fin de
    phrase, les silences internes, la ponctuation faible, l'équilibre des
    deux moitiés — et évite les coupures qui créeraient une moitié trop
    courte pour être affichée à la durée minimale.
    """
    words = sub.words
    n = len(words)
    min_s = duree_min_ms / 1000.0
    best_cut, best_score = n // 2, float("-inf")
    for i in range(1, n):
        prev = words[i - 1].text
        punct = 0.0
        if prev:
            if prev[-1] in ".!?…":
                punct = 2.0
            elif prev[-1] in ",;:":
                punct = 1.0
        gap = min(max(words[i].start - words[i - 1].end, 0.0), 1.0)
        balance = -abs(i - n / 2) / n
        duration_penalty = 0.0
        if (words[i - 1].end - words[0].start) < min_s or (
            words[-1].end - words[i].start
        ) < min_s:
            duration_penalty = -3.0
        score = 2.0 * gap + punct + balance + duration_penalty
        if score > best_score:
            best_cut, best_score = i, score
    first = Subtitle(words=words[:best_cut], start=words[0].start, end=words[best_cut - 1].end)
    second = Subtitle(words=words[best_cut:], start=words[best_cut].start, end=words[-1].end)
    return first, second


def _merge_short_rendered(
    subtitles: list[Subtitle],
    constraints,
    fmt: FormatConfig,
    measurer: FontMeasurer,
    available: float,
) -> list[Subtitle]:
    """Après le reflow, refusionne les sous-titres trop courts pour atteindre
    leur durée requise, à condition que le résultat fusionné tienne toujours
    dans le cadre (revalidation par rendu réel) — les scissions de l'étape 4
    ne doivent pas réintroduire de flashs illisibles.
    """
    from segmentation import _can_reach_needed

    changed = True
    while changed and len(subtitles) > 1:
        changed = False
        for i in range(len(subtitles)):
            if _can_reach_needed(subtitles, i, constraints):
                continue
            for j in (i - 1, i):  # fusion avec le précédent, sinon le suivant
                if not (0 <= j and j + 1 < len(subtitles)):
                    continue
                merged_words = subtitles[j].words + subtitles[j + 1].words
                if (merged_words[-1].end - merged_words[0].start) * 1000.0 > constraints.duree_max_ms:
                    continue
                lines = _layout_by_width(
                    [w.text for w in merged_words],
                    measurer,
                    available,
                    fmt.lignes_max,
                    fmt.mots_max_par_ligne,
                    fmt.caracteres_max_par_ligne,
                )
                if lines is None:
                    continue
                merged = Subtitle(
                    words=merged_words,
                    lines=lines,
                    start=merged_words[0].start,
                    end=merged_words[-1].end,
                )
                subtitles[j : j + 2] = [merged]
                changed = True
                break
            if changed:
                break
    return subtitles


def validate_and_reflow(
    subtitles: list[Subtitle],
    cfg: PipelineConfig,
    fmt: FormatConfig,
    format_name: str,
    measurer: FontMeasurer | None = None,
) -> tuple[list[Subtitle], RenderReport]:
    """Boucle de validation par rendu réel jusqu'à convergence (CDC §4 étape 4).

    - largeur disponible = largeur_sequence_px * ratio_largeur_texte,
      réduite par la marge de sécurité de calibration ;
    - toute ligne qui déborde entraîne une redistribution des mots, puis si
      nécessaire une scission du sous-titre ;
    - garde-fou : un élément insécable plus large que le cadre arrête la
      boucle sur ce sous-titre avec un avertissement explicite (pas de boucle
      infinie, pas d'échec silencieux).
    """
    if measurer is None:
        measurer = FontMeasurer(cfg.police, fmt.taille_police)

    frame_width = fmt.largeur_sequence_px * fmt.ratio_largeur_texte
    available = frame_width / (1.0 + cfg.marge_securite_rendu)
    constraints = HardConstraints.from_config(cfg, fmt)

    report = RenderReport(
        format_name=format_name,
        frame_width_px=frame_width,
        available_width_px=available,
        font_size=fmt.taille_police,
        subtitles_in=len(subtitles),
    )

    queue: list[Subtitle] = list(subtitles)
    done: list[Subtitle] = []
    iterations: dict[int, int] = {}

    while queue:
        sub = queue.pop(0)

        # Garde-fou 4.7 : mot insécable plus large que le cadre.
        oversized = [w.text for w in sub.words if measurer.width(w.text) > available]
        if oversized:
            report.warnings.append(
                f"[{format_name}] Élément insécable trop large pour le cadre "
                f"({available:.0f}px à taille {fmt.taille_police:g}) dans le "
                f"sous-titre {_ts(sub.start)} → {_ts(sub.end)} : "
                + ", ".join(f"« {t} » ({measurer.width(t):.0f}px)" for t in oversized)
                + " — sous-titre conservé tel quel, vérifier manuellement dans Premiere."
            )
            sub.lines = _fallback_lines(sub, fmt)
            done.append(sub)
            continue

        lines = _layout_by_width(
            [w.text for w in sub.words],
            measurer,
            available,
            fmt.lignes_max,
            fmt.mots_max_par_ligne,
            fmt.caracteres_max_par_ligne,
        )
        if lines is not None:
            if lines != sub.lines:
                report.reflowed += 1
            sub.lines = lines
            done.append(sub)
            continue

        # Aucune mise en page ne tient : réduire le nombre de mots affichés
        # simultanément en créant un sous-titre supplémentaire.
        key = id(sub)
        iterations[key] = iterations.get(key, 0) + 1
        if len(sub.words) < 2 or iterations[key] > MAX_REFLOW_ITERATIONS:
            report.warnings.append(
                f"[{format_name}] Impossible de faire tenir le sous-titre "
                f"{_ts(sub.start)} → {_ts(sub.end)} dans {fmt.lignes_max} ligne(s) "
                f"de {available:.0f}px — conservé avec débordement possible."
            )
            sub.lines = _fallback_lines(sub, fmt)
            done.append(sub)
            continue

        first, second = _split_subtitle(sub, cfg.duree_min_ms)
        iterations[id(first)] = iterations[key]
        iterations[id(second)] = iterations[key]
        report.splits += 1
        # Revalider les deux moitiés (boucle jusqu'à convergence).
        queue.insert(0, second)
        queue.insert(0, first)

    # Les scissions peuvent avoir créé des sous-titres trop courts pour être
    # étendus : refusionner (avec revalidation de largeur) avant d'appliquer
    # les contraintes de durée / cps / écart sur le découpage final (§4.6).
    done = _merge_short_rendered(done, constraints, fmt, measurer, available)
    _enforce_timing(done, constraints)
    for sub in done:
        if sub.chars_per_second > constraints.car_sec_max + 0.5:
            report.warnings.append(
                f"[{format_name}] Vitesse de lecture {sub.chars_per_second:.1f} car/s "
                f"> {constraints.car_sec_max:g} sur le sous-titre "
                f"{_ts(sub.start)} → {_ts(sub.end)} (parole trop dense pour être "
                f"étendue sans chevaucher le sous-titre suivant)."
            )
        elif sub.duration_ms < constraints.duree_min_ms - 1:
            report.warnings.append(
                f"[{format_name}] Durée d'affichage {sub.duration_ms:.0f} ms "
                f"< {constraints.duree_min_ms:g} ms sur le sous-titre "
                f"{_ts(sub.start)} → {_ts(sub.end)} (impossible d'étendre sans "
                f"chevaucher le sous-titre suivant)."
            )

    report.subtitles_out = len(done)
    return done, report


def _fallback_lines(sub: Subtitle, fmt: FormatConfig) -> list[str]:
    """Mise en page de repli quand le rendu ne peut pas être garanti."""
    from segmentation import layout_lines

    lines = layout_lines(
        [w.text for w in sub.words], fmt.lignes_max, max(fmt.mots_max_par_ligne, 1)
    )
    return lines if lines else [" ".join(w.text for w in sub.words)]


def _ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def generate_calibration_lines(
    font_path: str,
    font_size: float,
    frame_width_px: float,
    out_path: str,
) -> list[tuple[str, float]]:
    """Phase de calibration (CDC §8) : génère des lignes-test de largeur
    calculée connue à importer dans Premiere, pour mesurer l'écart entre la
    largeur prédite (Pillow) et la largeur rendue (Premiere) et ajuster
    `marge_securite_rendu` / `ratio_largeur_texte`."""
    measurer = FontMeasurer(font_path, font_size)
    base = "Le rendu réel fait foi dans Premiere Pro"
    lines: list[tuple[str, float]] = []
    text = ""
    for word in (base + " " + base + " " + base).split():
        candidate = (text + " " + word).strip()
        w = measurer.width(candidate)
        lines.append((candidate, w))
        text = candidate
        if w > frame_width_px * 1.2:
            break
    from srt_export import write_srt

    subs = []
    t = 0.0
    for line, width in lines:
        subs.append(
            Subtitle(
                words=[Word(text=line, start=t, end=t + 3.0)],
                lines=[line, f"[largeur prédite : {width:.0f}px / cadre {frame_width_px:.0f}px]"],
                start=t,
                end=t + 3.0,
            )
        )
        t += 4.0
    write_srt(subs, out_path)
    return lines
