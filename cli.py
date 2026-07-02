"""Pilotage du moteur en ligne de commande (CDC §11).

Exemples :
    python cli.py video.mp4 --police Atkinson.ttf --taille-16-9 100 --taille-9-16 80
    python cli.py video.mp4 --police Atkinson.ttf --taille-16-9 90 --taille-9-16 70 \
        --replay-json video_transcription.json
    python cli.py --preset projet.json video.mp4
    python cli.py --calibration --police Atkinson.ttf --taille-16-9 100
"""

from __future__ import annotations

import argparse
import sys

from config import PipelineConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cli.py",
        description=(
            "Transcription Whisper + sous-titrage à rendu garanti pour "
            "Premiere Pro (SRT 16:9 et 9:16 en une exécution)."
        ),
    )
    p.add_argument("video", nargs="?", help="Fichier vidéo (ou audio) à transcrire")
    p.add_argument("--preset", help="Préréglage projet à charger (JSON)")
    p.add_argument("--save-preset", help="Sauvegarder les réglages effectifs dans ce fichier")
    p.add_argument("--out-dir", help="Dossier de sortie (défaut : dossier de la vidéo)")
    p.add_argument(
        "--replay-json",
        help="Rejouer les étapes 2-5 depuis une transcription sauvegardée (pas de retranscription)",
    )

    g = p.add_argument_group("transcription")
    g.add_argument("--moteur", dest="moteur", help="faster-whisper (défaut) | whisperx")
    g.add_argument("--modele", help="large-v3 (défaut) | distil-large-v3 | medium | …")
    g.add_argument("--device", choices=["cuda", "cpu"], help="cuda (défaut) | cpu (tests)")

    g = p.add_argument_group("correction LLM (étape 1bis, optionnelle)")
    g.add_argument("--correction", action="store_true", help="Activer la correction LLM")
    g.add_argument("--correction-backend", choices=["ollama", "api"])
    g.add_argument("--correction-modele", help="Modèle LLM selon le backend")
    g.add_argument("--glossaire", help="Fichier texte de noms propres / termes métier")
    g.add_argument("--ollama-url", help="URL du serveur Ollama local")
    g.add_argument("--api-key", help="Clé API pour le backend cloud")

    g = p.add_argument_group("rendu (les tailles varient selon le projet)")
    g.add_argument("--police", help="Chemin du fichier .ttf/.otf (Atkinson Hyperlegible)")
    g.add_argument("--taille-16-9", type=float, help="Taille de police du projet en 16:9")
    g.add_argument("--taille-9-16", type=float, help="Taille de police du projet en 9:16")
    g.add_argument("--ratio-16-9", type=float, help="Ratio zone de texte 16:9 (défaut 0.75)")
    g.add_argument("--ratio-9-16", type=float, help="Ratio zone de texte 9:16 (défaut 0.75)")
    g.add_argument("--largeur-16-9", type=int, help="Largeur de séquence 16:9 en px (défaut 1920)")
    g.add_argument("--largeur-9-16", type=int, help="Largeur de séquence 9:16 en px (défaut 1080)")
    g.add_argument("--mots-ligne-16-9", type=int, help="Mots max/ligne 16:9 (défaut 7)")
    g.add_argument("--mots-ligne-9-16", type=int, help="Mots max/ligne 9:16 (défaut 4)")
    g.add_argument("--lignes-max", type=int, help="Lignes max par sous-titre (défaut 2)")
    g.add_argument("--caracteres-ligne", type=int, help="Caractères max/ligne (indicatif, 0 = off)")
    g.add_argument("--marge", type=float, help="Marge de sécurité de rendu (défaut 0.04)")

    g = p.add_argument_group("contraintes temporelles")
    g.add_argument("--duree-min", type=int, help="Durée min d'affichage en ms (défaut 833)")
    g.add_argument("--duree-max", type=int, help="Durée max d'affichage en ms (défaut 7000)")
    g.add_argument("--ecart-min", type=int, help="Écart min entre sous-titres en ms (défaut 83)")
    g.add_argument("--car-sec", type=float, help="Vitesse de lecture max en car/s (défaut 20)")
    g.add_argument("--seuil-silence", type=int, help="Seuil de silence en ms (défaut 250)")

    p.add_argument("--srt-bom", action="store_true", help="Écrire le BOM UTF-8 dans les SRT")
    p.add_argument(
        "--calibration",
        action="store_true",
        help="Générer un SRT de lignes-test de largeur connue (phase de calibration, CDC §8)",
    )
    return p


def apply_args(cfg: PipelineConfig, args: argparse.Namespace) -> PipelineConfig:
    """Applique les options CLI par-dessus le préréglage / les défauts."""
    if args.moteur:
        cfg.moteur_transcription = args.moteur
    if args.modele:
        cfg.modele = args.modele
    if args.device:
        cfg.device = args.device
    if args.correction:
        cfg.correction_llm.active = True
    if args.correction_backend:
        cfg.correction_llm.backend = args.correction_backend
    if args.correction_modele:
        cfg.correction_llm.modele_llm = args.correction_modele
    if args.glossaire:
        cfg.correction_llm.glossaire = args.glossaire
    if args.ollama_url:
        cfg.correction_llm.ollama_url = args.ollama_url
    if args.api_key:
        cfg.correction_llm.api_key = args.api_key
    if args.police:
        cfg.police = args.police

    f169, f916 = cfg.formats["16-9"], cfg.formats["9-16"]
    if args.taille_16_9 is not None:
        f169.taille_police = args.taille_16_9
    if args.taille_9_16 is not None:
        f916.taille_police = args.taille_9_16
    if args.ratio_16_9 is not None:
        f169.ratio_largeur_texte = args.ratio_16_9
    if args.ratio_9_16 is not None:
        f916.ratio_largeur_texte = args.ratio_9_16
    if args.largeur_16_9 is not None:
        f169.largeur_sequence_px = args.largeur_16_9
    if args.largeur_9_16 is not None:
        f916.largeur_sequence_px = args.largeur_9_16
    if args.mots_ligne_16_9 is not None:
        f169.mots_max_par_ligne = args.mots_ligne_16_9
    if args.mots_ligne_9_16 is not None:
        f916.mots_max_par_ligne = args.mots_ligne_9_16
    if args.lignes_max is not None:
        f169.lignes_max = f916.lignes_max = args.lignes_max
    if args.caracteres_ligne is not None:
        f169.caracteres_max_par_ligne = f916.caracteres_max_par_ligne = args.caracteres_ligne
    if args.marge is not None:
        cfg.marge_securite_rendu = args.marge

    if args.duree_min is not None:
        cfg.duree_min_ms = args.duree_min
    if args.duree_max is not None:
        cfg.duree_max_ms = args.duree_max
    if args.ecart_min is not None:
        cfg.ecart_min_ms = args.ecart_min
    if args.car_sec is not None:
        cfg.car_sec_max = args.car_sec
    if args.seuil_silence is not None:
        cfg.seuil_silence_ms = args.seuil_silence
    if args.srt_bom:
        cfg.srt_bom = True
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = PipelineConfig.load_preset(args.preset) if args.preset else PipelineConfig()
    cfg = apply_args(cfg, args)

    if args.save_preset:
        cfg.save_preset(args.save_preset)
        print(f"Préréglage sauvegardé : {args.save_preset}")

    if args.calibration:
        from rendering import generate_calibration_lines

        if not cfg.police:
            print("Erreur : --police est requis pour la calibration.", file=sys.stderr)
            return 2
        for name, fmt in cfg.formats.items():
            out = f"calibration_{name}.srt"
            frame = fmt.largeur_sequence_px * fmt.ratio_largeur_texte
            generate_calibration_lines(cfg.police, fmt.taille_police, frame, out)
            print(
                f"[{name}] Lignes-test générées dans {out} "
                f"(cadre {frame:.0f}px à taille {fmt.taille_police:g}) — "
                "à importer dans Premiere pour ajuster marge/ratio."
            )
        return 0

    if not args.video:
        print("Erreur : fichier vidéo requis (ou --calibration).", file=sys.stderr)
        return 2

    from pipeline import run_pipeline

    result = run_pipeline(
        args.video,
        cfg,
        out_dir=args.out_dir,
        replay_json=args.replay_json,
        progress=lambda msg: print(msg, flush=True),
    )
    print()
    print(result.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
