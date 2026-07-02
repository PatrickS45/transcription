"""Interface locale (Gradio) — couche de saisie uniquement (CDC §11).

Expose tous les paramètres du §5, un bouton Générer avec progression et
rapport final (avertissements de l'étape 4.7 inclus), et les préréglages par
projet (sauvegarde/rechargement). Le moteur reste indépendant : app.py ne
fait qu'assembler une PipelineConfig et appeler pipeline.run_pipeline.

Lancement : python app.py  →  http://127.0.0.1:7860 (aucun déploiement serveur)
"""

from __future__ import annotations

from pathlib import Path

from config import CorrectionConfig, FormatConfig, PipelineConfig
from pipeline import run_pipeline


def _build_config(
    police,
    moteur,
    modele,
    device,
    corr_active,
    corr_backend,
    corr_modele,
    glossaire,
    ollama_url,
    api_key,
    taille_169,
    ratio_169,
    mots_169,
    largeur_169,
    taille_916,
    ratio_916,
    mots_916,
    largeur_916,
    lignes_max,
    car_ligne,
    duree_min,
    duree_max,
    ecart_min,
    car_sec,
    seuil_silence,
    marge,
    srt_bom,
) -> PipelineConfig:
    return PipelineConfig(
        moteur_transcription=moteur,
        modele=modele,
        device=device,
        police=police,
        correction_llm=CorrectionConfig(
            active=bool(corr_active),
            backend=corr_backend,
            modele_llm=corr_modele or "",
            glossaire=glossaire or "",
            ollama_url=ollama_url or "http://localhost:11434",
            api_key=api_key or "",
        ),
        formats={
            "16-9": FormatConfig(
                taille_police=float(taille_169),
                ratio_largeur_texte=float(ratio_169),
                mots_max_par_ligne=int(mots_169),
                lignes_max=int(lignes_max),
                caracteres_max_par_ligne=int(car_ligne),
                largeur_sequence_px=int(largeur_169),
            ),
            "9-16": FormatConfig(
                taille_police=float(taille_916),
                ratio_largeur_texte=float(ratio_916),
                mots_max_par_ligne=int(mots_916),
                lignes_max=int(lignes_max),
                caracteres_max_par_ligne=int(car_ligne),
                largeur_sequence_px=int(largeur_916),
            ),
        },
        duree_min_ms=int(duree_min),
        duree_max_ms=int(duree_max),
        ecart_min_ms=int(ecart_min),
        car_sec_max=float(car_sec),
        seuil_silence_ms=int(seuil_silence),
        marge_securite_rendu=float(marge),
        srt_bom=bool(srt_bom),
    )


def build_app():
    import gradio as gr

    defaults = PipelineConfig()
    d169, d916 = defaults.formats["16-9"], defaults.formats["9-16"]

    with gr.Blocks(title="Transcription & sous-titrage — rendu garanti Premiere") as demo:
        gr.Markdown(
            "# Transcription et sous-titrage automatisé\n"
            "Whisper local (GPU) → découpage sur pauses → **validation par rendu "
            "réel** → SRT 16:9 et 9:16 en une exécution."
        )

        with gr.Row():
            video = gr.File(label="Fichier vidéo / audio", type="filepath")
            police = gr.File(
                label="Police .ttf/.otf (Atkinson Hyperlegible)", type="filepath"
            )

        with gr.Accordion("Transcription", open=True):
            with gr.Row():
                moteur = gr.Dropdown(
                    ["faster-whisper", "whisperx"],
                    value=defaults.moteur_transcription,
                    label="Moteur",
                )
                modele = gr.Dropdown(
                    ["large-v3", "distil-large-v3", "medium", "small"],
                    value=defaults.modele,
                    label="Modèle (compromis vitesse/qualité)",
                    allow_custom_value=True,
                )
                device = gr.Dropdown(
                    ["cuda", "cpu"], value=defaults.device, label="Device"
                )
            replay_json = gr.File(
                label="Rejouer depuis une transcription sauvegardée (JSON) — optionnel, "
                "pas de retranscription",
                type="filepath",
            )

        with gr.Accordion("Correction LLM (étape 1bis — optionnelle)", open=False):
            corr_active = gr.Checkbox(value=False, label="Activer la correction LLM")
            with gr.Row():
                corr_backend = gr.Dropdown(
                    ["ollama", "api"], value="ollama", label="Backend (ollama = local)"
                )
                corr_modele = gr.Textbox(label="Modèle LLM", placeholder="mistral / claude-sonnet-5…")
            with gr.Row():
                glossaire = gr.File(
                    label="Glossaire (noms propres, un par ligne) — optionnel",
                    type="filepath",
                )
                ollama_url = gr.Textbox(
                    value="http://localhost:11434", label="URL Ollama"
                )
                api_key = gr.Textbox(
                    label="Clé API (backend cloud uniquement)", type="password"
                )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Format 16:9")
                taille_169 = gr.Number(
                    value=d169.taille_police,
                    label="Taille de police du projet (16:9) — à saisir à chaque exécution",
                )
                ratio_169 = gr.Slider(0.5, 1.0, value=d169.ratio_largeur_texte, step=0.01,
                                      label="Ratio zone de texte")
                mots_169 = gr.Slider(1, 12, value=d169.mots_max_par_ligne, step=1,
                                     label="Mots max / ligne")
                largeur_169 = gr.Number(value=d169.largeur_sequence_px,
                                        label="Largeur de séquence (px)")
            with gr.Column():
                gr.Markdown("### Format 9:16")
                taille_916 = gr.Number(
                    value=d916.taille_police,
                    label="Taille de police du projet (9:16) — indépendante du 16:9",
                )
                ratio_916 = gr.Slider(0.5, 1.0, value=d916.ratio_largeur_texte, step=0.01,
                                      label="Ratio zone de texte")
                mots_916 = gr.Slider(1, 12, value=d916.mots_max_par_ligne, step=1,
                                     label="Mots max / ligne")
                largeur_916 = gr.Number(value=d916.largeur_sequence_px,
                                        label="Largeur de séquence (px)")

        with gr.Accordion("Contraintes communes", open=False):
            with gr.Row():
                lignes_max = gr.Slider(1, 3, value=2, step=1, label="Lignes max / sous-titre")
                car_ligne = gr.Number(value=0, label="Caractères max / ligne (0 = off, indicatif)")
                seuil_silence = gr.Number(value=defaults.seuil_silence_ms,
                                          label="Seuil de silence (ms)")
                marge = gr.Number(value=defaults.marge_securite_rendu,
                                  label="Marge de sécurité de rendu")
            with gr.Row():
                duree_min = gr.Number(value=defaults.duree_min_ms, label="Durée min (ms)")
                duree_max = gr.Number(value=defaults.duree_max_ms, label="Durée max (ms)")
                ecart_min = gr.Number(value=defaults.ecart_min_ms, label="Écart min (ms)")
                car_sec = gr.Number(value=defaults.car_sec_max, label="Car/s max")
            srt_bom = gr.Checkbox(value=False, label="BOM UTF-8 dans les SRT")

        with gr.Accordion("Préréglages par projet", open=False):
            with gr.Row():
                preset_path = gr.Textbox(
                    label="Fichier de préréglage (.json)", placeholder="mon_projet.json"
                )
                save_btn = gr.Button("💾 Sauvegarder les réglages")
                load_btn = gr.Button("📂 Recharger")
            preset_status = gr.Markdown()

        run_btn = gr.Button("▶ Générer les sous-titres", variant="primary")
        report_box = gr.Textbox(label="Rapport", lines=14, interactive=False)
        out_files = gr.Files(label="Fichiers générés (SRT 16:9, SRT 9:16, JSON)")

        all_params = [
            police, moteur, modele, device,
            corr_active, corr_backend, corr_modele, glossaire, ollama_url, api_key,
            taille_169, ratio_169, mots_169, largeur_169,
            taille_916, ratio_916, mots_916, largeur_916,
            lignes_max, car_ligne, duree_min, duree_max, ecart_min, car_sec,
            seuil_silence, marge, srt_bom,
        ]

        def do_run(video_path, replay, *params, progress=gr.Progress()):
            if not video_path:
                raise gr.Error("Sélectionnez un fichier vidéo.")
            if not params[0]:
                raise gr.Error("Sélectionnez le fichier de police (.ttf/.otf).")
            cfg = _build_config(*params)
            messages: list[str] = []

            def notify(msg: str) -> None:
                messages.append(msg)
                progress(0, desc=msg)

            result = run_pipeline(video_path, cfg, replay_json=replay, progress=notify)
            files = [str(p) for p in result.srt_files.values()]
            if result.transcription_json:
                files.append(str(result.transcription_json))
            return "\n".join(messages) + "\n\n" + result.summary(), files

        run_btn.click(do_run, inputs=[video, replay_json, *all_params],
                      outputs=[report_box, out_files])

        def do_save(path, *params):
            if not path:
                return "Indiquez un chemin de fichier de préréglage."
            cfg = _build_config(*params)
            cfg.save_preset(path)
            return f"Préréglage sauvegardé : `{path}` (la clé API n'est jamais écrite)."

        save_btn.click(do_save, inputs=[preset_path, *all_params], outputs=[preset_status])

        def do_load(path):
            if not path or not Path(path).is_file():
                return [gr.update()] * len(all_params) + ["Fichier de préréglage introuvable."]
            cfg = PipelineConfig.load_preset(path)
            f169, f916 = cfg.formats["16-9"], cfg.formats["9-16"]
            values = [
                cfg.police or None, cfg.moteur_transcription, cfg.modele, cfg.device,
                cfg.correction_llm.active, cfg.correction_llm.backend,
                cfg.correction_llm.modele_llm, cfg.correction_llm.glossaire or None,
                cfg.correction_llm.ollama_url, "",
                f169.taille_police, f169.ratio_largeur_texte,
                f169.mots_max_par_ligne, f169.largeur_sequence_px,
                f916.taille_police, f916.ratio_largeur_texte,
                f916.mots_max_par_ligne, f916.largeur_sequence_px,
                f169.lignes_max, f169.caracteres_max_par_ligne,
                cfg.duree_min_ms, cfg.duree_max_ms, cfg.ecart_min_ms, cfg.car_sec_max,
                cfg.seuil_silence_ms, cfg.marge_securite_rendu, cfg.srt_bom,
            ]
            return values + [f"Préréglage chargé : `{path}`."]

        load_btn.click(do_load, inputs=[preset_path],
                       outputs=all_params + [preset_status])

    return demo


if __name__ == "__main__":
    build_app().launch()  # servi en local uniquement
