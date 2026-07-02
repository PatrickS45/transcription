# Outil de transcription et sous-titrage automatisé

Implémentation du [cahier des charges v2](cahierdeschargestranscriptionv2.md) :
transcription Whisper locale (GPU), découpage des sous-titres sur les pauses
naturelles de la parole (VAD), **validation par calcul de rendu réel**
(aucune ligne ne déborde dans Adobe Premiere Pro), et export SRT en deux
formats (16:9 et 9:16) en une seule exécution.

## Installation

```bash
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
# Pour le GPU (RTX 4080 Super) : installer la build CUDA de torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Fournir la police du projet (Atkinson Hyperlegible, `.ttf`/`.otf`).

## Utilisation

### Interface locale (Gradio)

```bash
python app.py
# → http://127.0.0.1:7860 — tous les paramètres du CDC §5 sont réglables,
#   préréglages par projet sauvegardables/rechargeables.
```

### Ligne de commande

```bash
# Pipeline complet : transcription + 2 SRT
python cli.py video.mp4 --police AtkinsonHyperlegible-Regular.ttf \
    --taille-16-9 100 --taille-9-16 80

# Itérer sur la mise en page sans retranscrire (rejeu depuis le JSON)
python cli.py video.mp4 --police AtkinsonHyperlegible-Regular.ttf \
    --taille-16-9 90 --taille-9-16 72 \
    --replay-json video_transcription.json

# Correction LLM optionnelle (locale via Ollama, avec glossaire)
python cli.py video.mp4 --police Atkinson.ttf --taille-16-9 100 --taille-9-16 80 \
    --correction --correction-backend ollama --correction-modele mistral \
    --glossaire glossaire.txt

# Préréglages par projet
python cli.py video.mp4 --preset mon_projet.json
python cli.py video.mp4 --police Atkinson.ttf --taille-16-9 100 --taille-9-16 80 \
    --save-preset mon_projet.json

# Phase de calibration (CDC §8) : lignes-test de largeur connue à importer
# dans Premiere pour ajuster --marge / --ratio-16-9 / --ratio-9-16
python cli.py --calibration --police Atkinson.ttf --taille-16-9 100 --taille-9-16 80
```

Sorties : `{nom_video}_16-9.srt`, `{nom_video}_9-16.srt` (UTF-8, BOM optionnel
via `--srt-bom`) et `{nom_video}_transcription.json` (format intermédiaire
rejouable).

## Architecture (CDC §11)

```
app.py               → interface locale (Gradio) — couche de saisie uniquement
cli.py               → même moteur en ligne de commande
pipeline.py          → orchestration des étapes 1 → 5
models.py            → format intermédiaire commun (segments + mots + timestamps)
transcription/       → backends interchangeables (faster-whisper défaut, whisperx)
correction.py        → étape 1bis optionnelle (Ollama / API), préserve les timestamps
segmentation.py      → découpage sur pauses + contraintes dures (pur Python, sans GPU)
rendering.py         → mesure de largeur Pillow + reflow + garde-fous (pur Python)
srt_export.py        → écriture des deux fichiers SRT
config.py            → paramètres, valeurs par défaut, préréglages par projet
```

## Principes clés

- **Rendu réel** : chaque ligne est mesurée en pixels (Pillow/FreeType) à la
  taille de police fournie **à chaque exécution** (jamais codée en dur),
  comparée à `largeur_sequence × ratio_largeur_texte` (défaut 0.75, mesuré
  dans Premiere), avec une marge de sécurité réglable (défaut 4 %). Les
  sous-titres qui débordent sont redistribués puis scindés jusqu'à
  convergence ; un mot insécable trop large produit un avertissement
  explicite dans le rapport (jamais d'échec silencieux).
- **Pauses naturelles** : coupures alignées sur les silences VAD
  (seuil réglable, défaut 250 ms) ; en cas de conflit, les contraintes
  dures (durée 833–7000 ms, 20 car/s, mots/ligne, 2 lignes max, écart 83 ms)
  priment.
- **Moteur interchangeable** : tout backend qui restitue le format
  intermédiaire (JSON, timestamps au niveau du mot) s'intègre sans refonte.
- **Langue forcée `fr`**, jamais d'auto-détection.

## Tests

```bash
python -m pytest tests/ -q
```

Les étapes 2 à 5 (segmentation, rendu, export) sont en pur Python et
entièrement testées sans GPU. La transcription (étape 1) nécessite
`faster-whisper` + CUDA sur la machine cible.
