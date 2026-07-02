"""Étape 1bis — correction du texte par LLM (optionnelle, désactivée par défaut).

Le LLM reçoit le texte transcrit et retourne un texte corrigé ; il ne touche
jamais aux timestamps. La correction s'applique segment par segment et la
correspondance texte/temps est reconstruite de manière déterministe si le
LLM fusionne ou scinde des mots. Hors chemin critique : le pipeline complet
fonctionne à l'identique quand cette étape est désactivée.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Callable

from config import CorrectionConfig
from models import Segment, TranscriptionResult, Word

SYSTEM_PROMPT = (
    "Tu corriges des transcriptions automatiques en français : orthographe, "
    "ponctuation, noms propres et jargon. Tu ne reformules pas, tu ne résumes "
    "pas, tu n'ajoutes ni ne retires d'information. Tu réponds uniquement avec "
    "le texte corrigé, sans commentaire."
)


def load_glossary(path: str) -> list[str]:
    """Glossaire utilisateur : un terme/nom propre par ligne, fichier texte simple."""
    if not path:
        return []
    text = Path(path).read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def build_prompt(text: str, glossary: list[str]) -> str:
    prompt = ""
    if glossary:
        prompt += (
            "Référence de correction — noms propres et termes métier à "
            "orthographier exactement ainsi :\n"
            + "\n".join(f"- {term}" for term in glossary)
            + "\n\n"
        )
    prompt += f"Texte à corriger :\n{text}"
    return prompt


def realign_words(original: list[Word], corrected_text: str) -> list[Word]:
    """Reconstruit l'alignement texte/temps de manière déterministe.

    - même nombre de mots : correspondance 1:1, timestamps inchangés ;
    - nombre différent (fusion/scission par le LLM) : les nouveaux mots sont
      répartis proportionnellement à leur position en caractères dans le
      texte, interpolée sur les frontières temporelles des mots d'origine.
    Les timestamps de début et de fin du segment sont toujours préservés.
    """
    new_texts = corrected_text.split()
    if not new_texts or not original:
        return list(original)
    if len(new_texts) == len(original):
        return [
            Word(text=t, start=w.start, end=w.end)
            for t, w in zip(new_texts, original)
        ]

    seg_start, seg_end = original[0].start, original[-1].end
    total_chars = sum(len(t) for t in new_texts) + (len(new_texts) - 1)

    # Table de correspondance position-caractère → temps, construite sur les
    # mots d'origine (répartition proportionnelle à la longueur des mots).
    orig_total = sum(len(w.text) for w in original) + (len(original) - 1)

    def char_to_time(pos: float) -> float:
        frac = pos / max(total_chars, 1)
        target = frac * orig_total
        acc = 0.0
        for w in original:
            span = len(w.text) + 1
            if target <= acc + span or w is original[-1]:
                inner = min(max((target - acc) / span, 0.0), 1.0)
                return w.start + inner * (w.end - w.start)
            acc += span
        return seg_end

    words: list[Word] = []
    pos = 0
    for i, t in enumerate(new_texts):
        start = seg_start if i == 0 else char_to_time(pos)
        pos += len(t)
        end = seg_end if i == len(new_texts) - 1 else char_to_time(pos)
        pos += 1  # espace
        if end < start:
            end = start
        words.append(Word(text=t, start=start, end=end))
    return words


# --------------------------------------------------------------------- backends


def _correct_ollama(text: str, cfg: CorrectionConfig) -> str:
    """LLM local via Ollama — aucune donnée ne quitte la machine."""
    payload = json.dumps(
        {
            "model": cfg.modele_llm or "mistral",
            "system": SYSTEM_PROMPT,
            "prompt": text,
            "stream": False,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        cfg.ollama_url.rstrip("/") + "/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))["response"].strip()


def _correct_api(text: str, cfg: CorrectionConfig) -> str:
    """API cloud (Claude) — à n'utiliser que si la confidentialité le permet."""
    try:
        import anthropic
    except ImportError as e:  # pragma: no cover - dépend de l'environnement
        raise RuntimeError(
            "Le SDK anthropic n'est pas installé : pip install anthropic"
        ) from e
    if not cfg.api_key:
        raise RuntimeError("Clé API requise pour le backend cloud (correction_llm.api_key).")
    client = anthropic.Anthropic(api_key=cfg.api_key)
    message = client.messages.create(
        model=cfg.modele_llm or "claude-sonnet-5",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": text}],
    )
    return message.content[0].text.strip()


_BACKENDS: dict[str, Callable[[str, CorrectionConfig], str]] = {
    "ollama": _correct_ollama,
    "api": _correct_api,
}


def correct_transcription(
    result: TranscriptionResult,
    cfg: CorrectionConfig,
    progress: Callable[[str], None] | None = None,
) -> TranscriptionResult:
    """Corrige segment par segment en préservant l'alignement texte/temps."""
    if not cfg.active:
        return result
    if cfg.backend not in _BACKENDS:
        raise ValueError(
            f"Backend de correction inconnu : {cfg.backend!r} "
            f"(disponibles : {', '.join(sorted(_BACKENDS))})"
        )
    notify = progress or (lambda msg: None)
    glossary = load_glossary(cfg.glossaire)
    corrector = _BACKENDS[cfg.backend]

    corrected = TranscriptionResult(
        language=result.language,
        engine=result.engine,
        model=result.model,
        audio_duration=result.audio_duration,
    )
    for i, seg in enumerate(result.segments, start=1):
        notify(f"Correction LLM du segment {i}/{len(result.segments)}…")
        new_text = corrector(build_prompt(seg.text, glossary), cfg)
        # Réponse vide ou aberrante : on conserve le segment d'origine.
        if not new_text.strip():
            corrected.segments.append(seg)
            continue
        corrected.segments.append(Segment(words=realign_words(seg.words, new_text)))
    return corrected
