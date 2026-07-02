# Cahier des charges technique — Outil de transcription et sous-titrage automatisé

> **Version 2 — 02/07/2026.** Amendements par rapport à la v1 : moteur de transcription interchangeable (§4 étape 1), correction optionnelle par LLM (§4 étape 1bis), interface locale de saisie des paramètres actée (§9 révisé, nouveau §11), tous les paramètres ouverts exposés comme variables réglables dans l'interface (§5).

## 1. Objectif

Développer un outil Python autonome qui transcrit une vidéo en français avec Whisper (local, GPU), segmente les sous-titres en respectant les pauses naturelles de la parole, applique des contraintes strictes de mise en forme (mots/ligne, lignes/sous-titre, caractères/ligne), et **garantit par calcul de rendu réel** qu'aucune ligne ne débordera visuellement dans Adobe Premiere Pro, pour deux formats de sortie (16:9 et 9:16).

Le script remplace un workflow manuel actuel (Subtitle Edit + import SRT + retouche dans Premiere) qui échoue à garantir le rendu final, et un plugin commercial testé (Echoe Scribe) qui ne respecte pas les pauses naturelles.

## 2. Contexte matériel et environnement

- OS : Windows
- GPU : NVIDIA RTX 4080 Super (CUDA disponible)
- Langue source : français (à forcer explicitement, jamais auto-détection)
- Logiciel de montage cible : Adobe Premiere Pro (import SRT)
- Police utilisée : Atkinson Hyperlegible (fichier .ttf/.otf à fournir au script)
- Taille de police cible dans Premiere : **variable selon le projet**, pas de valeur fixe. 100 est un exemple observé, pas une constante. Le script doit accepter la taille en paramètre à chaque exécution, indépendamment pour le 16:9 et pour le 9:16 (les deux formats peuvent avoir des tailles différentes sur un même projet)

## 3. Formats de sortie requis

Le script doit produire **deux fichiers SRT distincts** à partir d'une seule transcription :

| Format | Ratio | Usage |
|---|---|---|
| Horizontal | 16:9 | format standard |
| Vertical | 9:16 | réseaux sociaux |

Chaque format a sa propre largeur de cadre disponible, donc ses propres contraintes de mise en page (voir §6). Les deux fichiers doivent être générés en une seule exécution du script, à partir de la même transcription source (pas de retranscription).

## 4. Pipeline de traitement

### Étape 1 — Transcription (moteur interchangeable)

Le moteur de transcription est une **brique interchangeable** derrière une interface commune, pas un choix figé dans le code. Toute implémentation doit fournir en sortie le même format intermédiaire : liste de segments avec texte + timestamps début/fin, **avec timestamps au niveau du mot**.

- **Moteur par défaut : faster-whisper, modèle large-v3** (local, licence libre, VAD intégré)
- Device : CUDA (GPU), avec repli CPU possible (lent, pour tests uniquement)
- Langue : forcée sur `fr` (pas d'auto-détection)
- Le **modèle est un paramètre** (`large-v3` par défaut, mais `distil-large-v3`, `medium`, etc. sélectionnables — compromis vitesse/qualité au choix de l'utilisateur)
- **Moteurs alternatifs prévus par l'architecture** (implémentables sans refonte, pas nécessairement livrés en v1) :
  - WhisperX (meilleur alignement des timestamps mot par mot via alignement forcé)
  - Tout futur modèle ASR local ou API cloud de transcription, tant qu'il restitue le format intermédiaire commun
- La sortie intermédiaire de l'étape 1 est **sauvegardée sur disque (JSON)** : elle sert de source unique aux deux formats (§3) et permet de rejouer les étapes 2 à 5 sans retranscrire (itération rapide sur les paramètres de mise en page)

### Étape 1bis — Correction du texte par LLM (optionnelle, désactivée par défaut)

Post-traitement facultatif de la transcription pour améliorer la fidélité du texte (noms propres, jargon, ponctuation — critère de validation n°5) :

- Le LLM reçoit le texte transcrit et retourne un texte corrigé ; il ne touche **jamais** aux timestamps. La correction s'applique mot à mot ou segment par segment de façon à préserver l'alignement texte/temps (si le LLM fusionne ou scinde des mots, la correspondance avec les timestamps mot doit être reconstruite de manière déterministe)
- Deux backends prévus, sélectionnables par paramètre :
  - **LLM local** via Ollama (aucune donnée ne quitte la machine)
  - **API cloud** (ex. API Claude) — nécessite une clé API fournie par l'utilisateur, à n'utiliser que si la confidentialité du contenu le permet
- Un **glossaire utilisateur** (liste de noms propres/termes métier, fichier texte simple) est passé au LLM comme référence de correction
- Cette étape est **hors chemin critique** : le pipeline complet doit fonctionner à l'identique quand elle est désactivée

### Étape 2 — Détection des pauses (VAD)
- Utiliser la détection de silence déjà intégrée à faster-whisper (VAD basé sur Silero, disponible nativement via l'option `vad_filter=True` de faster-whisper) pour identifier les silences dans l'audio. Si un moteur alternatif ne fournit pas de VAD, le module VAD (Silero) est appelé séparément — la détection de pauses fait partie du contrat du format intermédiaire
- Les limites de sous-titres (découpage en segments affichables) doivent s'aligner sur les pauses détectées : ne jamais couper un sous-titre au milieu d'un groupe de mots séparé par un silence court, et privilégier une coupure de sous-titre sur un silence plutôt qu'au milieu d'un flux continu de parole
- Seuil de silence à exposer comme paramètre configurable (durée minimale en ms pour qu'un silence soit considéré comme point de coupure valide)

### Étape 3 — Segmentation initiale en sous-titres
Contraintes dures à appliquer sur chaque sous-titre généré, **paramétrables par format** (16:9 / 9:16) :

- Nombre de mots maximum par ligne
- Nombre de lignes maximum par sous-titre (défaut : 2)
- Nombre de caractères maximum par ligne
- Durée minimale d'affichage (ms) — défaut 833ms (référence Netflix)
- Durée maximale d'affichage (ms) — défaut 7000ms
- Écart minimum entre deux sous-titres consécutifs (ms) — défaut 83ms
- Vitesse de lecture max (caractères/seconde) — défaut 20 car/s

En cas de conflit entre respect de la pause naturelle et respect des contraintes dures (mots/lignes/caractères), les contraintes dures sont prioritaires — mieux vaut couper légèrement hors d'une pause idéale que produire un sous-titre illisible ou hors normes.

### Étape 4 — Validation par rendu réel (PIÈCE MAÎTRESSE)

**Problème à résoudre** : un compte de caractères ou de mots ne garantit pas qu'une ligne tienne visuellement à l'écran, car la largeur réelle dépend de la police, de la **taille de police du projet en cours** (variable — voir §2, ne jamais coder une taille en dur), et de la largeur propre à chaque caractère (un "i" et un "m" n'occupent pas le même espace).

**Méthode obligatoire** :
1. Charger la police Atkinson Hyperlegible (fichier fourni) via une librairie de rendu de texte (ex. Pillow / PIL avec `ImageFont.truetype`)
2. La **taille de police est un paramètre d'entrée fourni à chaque exécution du script**, distinct pour le 16:9 et pour le 9:16 (deux valeurs indépendantes, potentiellement différentes)
3. Pour chaque ligne de chaque sous-titre généré à l'étape 3, calculer sa largeur réelle en pixels **à la taille de police fournie pour cette exécution** — jamais une valeur par défaut supposée
4. Comparer cette largeur à la largeur de cadre disponible en pixels, spécifique à chaque format :
   - 16:9 : largeur de cadre à définir (mesurée dans Premiere)
   - 9:16 : largeur de cadre à mesurer séparément (cadre plus étroit)
5. **Si une ligne dépasse la largeur de cadre** : reformater automatiquement le sous-titre concerné — redistribuer les mots entre les lignes disponibles, ou si nécessaire réduire le nombre de mots affichés simultanément (créer un sous-titre supplémentaire), jusqu'à ce que chaque ligne tienne réellement dans l'espace disponible, compte tenu de la taille de police de ce projet précis
6. Cette validation doit boucler jusqu'à convergence : après reformatage, revérifier que le nouveau découpage respecte toujours les contraintes dures de l'étape 3 (durée min/max, car/s, etc.)
7. **Garde-fou de convergence** : si un élément insécable (mot unique, URL, nom composé) dépasse à lui seul la largeur de cadre, la boucle s'arrête sur ce sous-titre avec un avertissement explicite dans le rapport de sortie (pas de boucle infinie, pas d'échec silencieux)
8. **Marge de sécurité de calibration** : un facteur de marge paramétrable (défaut ex. 3-5%) est appliqué à la largeur mesurée, pour absorber les écarts éventuels entre le rendu Pillow/FreeType et le moteur de rendu de Premiere. À ajuster lors de la phase de calibration (§8)

**Ce mécanisme doit tourner indépendamment pour chaque format (16:9 et 9:16)**, avec sa propre taille de police et sa propre largeur de cadre, car un même segment de texte peut tenir en 1 ligne en 16:9 et nécessiter 2 lignes en 9:16 — et ces tailles varient elles-mêmes d'un projet à l'autre.

### Étape 5 — Export
- Génère deux fichiers `.srt` :
  - `{nom_video}_16-9.srt`
  - `{nom_video}_9-16.srt`
- Encodage UTF-8 (obligatoire pour les accents français ; présence du BOM à valider contre le comportement d'import de Premiere lors des premiers tests)
- Format SRT standard (numérotation, timestamps `HH:MM:SS,mmm` avec virgule, texte, ligne vide de séparation)

## 5. Paramètres à exposer

**Principe général** : tous les paramètres ci-dessous sont des **variables réglables depuis l'interface** (§11) à chaque exécution, avec des valeurs par défaut sensées. Aucune valeur "à définir" n'est figée dans le code : les valeurs indicatives ci-dessous sont des défauts de départ, modifiables en front sans toucher au code. Les réglages d'un projet sont sauvegardables/rechargeables (préréglages par projet).

```
- langue : fr (fixe)
- moteur_transcription : faster-whisper (défaut) | whisperx | [extensible]
- modele : large-v3 (défaut) | distil-large-v3 | medium | ...
- correction_llm :
    active : false (défaut)
    backend : ollama (local) | api (ex. Claude)
    modele_llm : [au choix selon backend]
    glossaire : chemin vers fichier de termes/noms propres (optionnel)
- police : chemin vers le fichier .ttf/.otf
- formats :
    16-9 :
      taille_police : [fournie à chaque exécution, ex. 100 — variable selon le projet]
      ratio_largeur_texte : 0.75 (défaut, réglable)
      mots_max_par_ligne : 7 (défaut de départ, réglable)
      lignes_max : 2
      caracteres_max_par_ligne : [indicatif — le rendu réel fait foi]
    9-16 :
      taille_police : [fournie à chaque exécution — indépendante du 16:9]
      ratio_largeur_texte : 0.75 (défaut, réglable)
      mots_max_par_ligne : 4 (défaut de départ, réglable)
      lignes_max : 2
      caracteres_max_par_ligne : [indicatif — le rendu réel fait foi]
- duree_min_ms : 833
- duree_max_ms : 7000
- ecart_min_ms : 83
- car_sec_max : 20
- seuil_silence_ms : 250 (défaut de départ, réglable)
- marge_securite_rendu : 0.04 (défaut de départ, réglable — voir §4 étape 4.8)
```

**Important** : `taille_police` n'est jamais une constante codée en dur dans le script — c'est une valeur saisie à chaque exécution (interface ou ligne de commande), car elle varie selon la nature du projet et peut différer entre le 16:9 et le 9:16 d'un même projet.

## 6. Mesures préalables nécessaires (à faire avant/pendant le développement)

Deux types de mesures, à ne pas confondre :

**Fixe par format** (à mesurer une fois, ne change pas d'un projet à l'autre) :
- Largeur en pixels de la zone de texte des sous-titres en 16:9, à la résolution de référence (ex. 1920x1080)
  - **Mesure obtenue par capture d'écran (Program Monitor) le 02/07/2026** : la zone de texte occupe environ **75% de la largeur totale du cadre vidéo**. Pour une séquence 1920x1080, cela donne une largeur de zone de texte d'environ **1440 px**.
  - Cette mesure est dérivée d'un ratio (zone de texte / largeur totale de cadre) plutôt que d'une valeur absolue lue dans les propriétés Premiere (non trouvée dans l'interface disponible) — à considérer comme une estimation de départ, à affiner si le rendu réel diverge après les premiers tests du script.
  - **Le script doit accepter ce paramètre en pourcentage de la largeur de séquence (ex. 0.75) plutôt qu'en pixels fixes**, pour rester valide quelle que soit la résolution exacte du projet.
- Largeur en pixels de la zone de texte des sous-titres en 9:16, à la résolution de référence (ex. 1080x1920)
  - **Mesure obtenue par capture d'écran (Program Monitor) le 02/07/2026** : la zone de texte occupe environ **75-76% de la largeur totale du cadre vidéo** — un ratio quasiment identique à celui mesuré en 16:9. Pour une séquence 1080x1920, cela donne une largeur de zone de texte d'environ **810-820 px**.
  - Comme pour le 16:9, cette valeur est une estimation par ratio dérivée d'une capture d'écran, à valider lors des premiers tests du script.

**Variable par projet** (fournie à chaque exécution, jamais mesurée une fois pour toutes) :
- Taille de police utilisée dans le projet en cours, pour le 16:9
- Taille de police utilisée dans le projet en cours, pour le 9:16

Le calcul pixel-réel (§4) combine la largeur de cadre fixe (exprimée en ratio de la largeur de séquence) avec la taille de police variable à chaque exécution — c'est cette combinaison qui détermine si une ligne tient ou non, pas une valeur figée.

**Note méthodologique** : la valeur de 75% ci-dessus a été estimée visuellement à partir d'une capture d'écran (mesure de rectangle en pixels sur l'image, rapportée à la largeur totale visible). Ce n'est pas une lecture d'un champ numérique exact dans Premiere. Il est recommandé de la traiter comme un point de départ et de la valider/corriger lors des premiers tests du script sur une séquence réelle.

**Synthèse** : les deux formats (16:9 et 9:16) montrent un ratio de zone de texte quasiment identique (~75%), ce qui suggère une marge de sécurité proportionnelle appliquée uniformément par Premiere plutôt qu'une valeur fixe en pixels. Le script peut donc utiliser un seul paramètre `ratio_largeur_texte` par défaut (0.75), applicable aux deux formats, sauf si une vérification ultérieure révèle une différence.

## 7. Dépendances techniques prévisibles

- Python 3.10+
- `faster-whisper` (transcription + VAD intégré) — moteur par défaut
- `torch` avec support CUDA (accélération GPU)
- `Pillow` (calcul de largeur de texte rendu)
- Police Atkinson Hyperlegible au format `.ttf` ou `.otf`
- Interface locale : `gradio` (ou équivalent léger servi en local) — voir §11
- Optionnel (étape 1bis) : client Ollama (LLM local) et/ou SDK du fournisseur d'API choisi (ex. `anthropic`)
- Optionnel (moteur alternatif) : `whisperx`

## 8. Critères de validation / tests

Le script est considéré fonctionnel si, sur un échantillon de vidéos test :

1. Aucune ligne de sous-titre ne déborde visuellement une fois importée dans Premiere à la taille de police du projet, sur les deux formats
2. Les coupures de sous-titres coïncident visiblement avec les pauses naturelles de la parole à l'écoute
3. Aucun sous-titre ne dépasse les contraintes dures définies (durée, car/s, mots/ligne, lignes max)
4. Les deux fichiers SRT (16:9, 9:16) sont générés en une seule exécution, sans retranscription redondante
5. Le texte transcrit est fidèle à l'audio (vérification qualitative sur quelques passages, en particulier noms propres et jargon)

**Phase de calibration préalable** : avant validation, générer des lignes-test de largeur calculée connue, les importer dans Premiere, mesurer l'écart entre la largeur prédite (Pillow) et la largeur rendue (Premiere), et ajuster `marge_securite_rendu` et/ou `ratio_largeur_texte` en conséquence. Cette calibration est faite une fois par police/format et documentée.

## 9. Hors périmètre (pour cette version)

- Pas d'intégration directe dans Premiere (pas d'extension UXP) — le livrable produit des fichiers SRT à importer manuellement. *(Décision confirmée après échange : une extension n'aurait de toute façon pas accès au GPU/Python et serait fragile aux mises à jour de Premiere.)*
- ~~Pas d'interface graphique dans la première version~~ **Révisé** : une interface locale légère de saisie des paramètres fait partie du périmètre — voir §11. Le moteur reste également pilotable en ligne de commande.
- Pas de gestion de la diarisation (identification des locuteurs) sauf besoin exprimé ultérieurement
- Pas de style/couleur/position embarqués dans le SRT (le stylage reste géré dans Premiere via Track Style, le format SRT ne portant pas cette information)
- Pas de prévisualisation visuelle des sous-titres rendus dans l'interface (candidate pour une v2)

## 10. Historique du problème (contexte pour le développeur / Claude Code)

- Le workflow initial (Subtitle Edit → export SRT → import Premiere) a échoué car Subtitle Edit calibre son découpage en nombre de caractères, sans connaître la largeur réelle de rendu dans Premiere à la taille de police utilisée (100) — désynchronisation entre les deux outils
- Le plugin commercial testé (Echoe Scribe) transcrit et reflow correctement mais ne respecte pas les pauses naturelles de la parole (découpage purement arithmétique par nombre de mots/caractères, sans VAD)
- D'où l'exigence combinée de ce cahier des charges : respect des pauses (VAD) **et** garantie de rendu par calcul pixel réel, ce qu'aucun des deux outils testés ne fait simultanément

## 11. Interface utilisateur et architecture logicielle

### Interface locale (front)

Une interface web locale légère (ex. Gradio, servie sur la machine de l'utilisateur, aucun déploiement serveur) expose **tous les paramètres du §5** :

- Sélection du fichier vidéo et du fichier de police
- Choix du moteur de transcription et du modèle
- Activation/configuration de la correction LLM (backend, glossaire)
- Tailles de police 16:9 et 9:16, ratios de largeur, mots/ligne, seuils de durée et de silence, marge de sécurité
- Bouton "Générer" + affichage de la progression et du rapport final (avertissements de l'étape 4.7 inclus)
- **Préréglages par projet** : sauvegarde et rechargement des réglages (fichier de config par projet)

### Architecture (moteur découplé)

Le moteur est indépendant de l'interface : l'interface n'est qu'une couche de saisie au-dessus d'un cœur pur Python, également utilisable en ligne de commande.

```
app.py               → interface locale (Gradio) — couche de saisie uniquement
cli.py               → même moteur en ligne de commande
├── transcription/   → backends interchangeables (faster-whisper par défaut,
│                      whisperx…, contrat de sortie commun : segments + mots + timestamps)
├── correction.py    → étape 1bis optionnelle (Ollama / API), préserve les timestamps
├── segmentation.py  → découpage sur pauses + contraintes dures (pur Python, testable sans GPU)
├── rendering.py     → mesure de largeur Pillow + reflow + garde-fous (pur Python, testable sans GPU)
├── srt_export.py    → écriture des deux fichiers SRT
└── config.py        → paramètres, valeurs par défaut, préréglages par projet
```

Un mode "rejouer depuis la transcription sauvegardée" (JSON de l'étape 1) permet d'itérer sur les paramètres de mise en page sans retranscrire.
