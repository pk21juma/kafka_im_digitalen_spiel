import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from .utils import build_tables

def embedding_evaluation(games, kafka_themes, model=SentenceTransformer('all-mpnet-base-v2'), batch_size=16, device: str | None = None):
    """
    Compute embedding-based similarity scores between game texts and kafka_themes.

    Produces per-game scores for each theme (0-100) and an overall "Embedding_Total".
    Results are passed to utils.build_tables(...) which writes CSVs and Top-10 files.

    Args:
        games (list[dict]): Spieleliste (je dict mindestens "app_id", "title", "description", "achievements").
        kafka_themes (dict): Mapping theme -> definition/text.
        model: SentenceTransformer model (already loaded).
        batch_size (int): Batch size for encoding.
        device (str|None): "cuda" or "cpu". If None, autodetected via torch.cuda.is_available().
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    texts = []
    game_ids = []
    for game in games:
        achievement_texts = []
        for ach in game.get("achievements", []):
            if isinstance(ach, dict):
                achievement_texts.append(ach.get("title", ""))
                achievement_texts.append(ach.get("description", ""))
            elif isinstance(ach, str):
                achievement_texts.append(ach)
        full_text = (game.get("description", "") or "") + " " + " ".join(achievement_texts)
        texts.append(full_text)
        game_ids.append(str(game["app_id"]))

    theme_embeddings = {}
    theme_weights = {}
    for theme, definition in kafka_themes.items():
        if theme in ["kafkaesque", "absurd", "meritocracy", "civil service"]:
            weight = 5
        elif theme in ["alienated", "bureaucratic", "isolated"]:
            weight = 3
        else:
            weight = 1
        emb = model.encode(definition, convert_to_tensor=True, device=device)
        theme_embeddings[theme] = emb
        theme_weights[theme] = weight

    all_scores = {}
    num_games = len(games)

    for start_idx in range(0, num_games, batch_size):
        end_idx = min(start_idx + batch_size, num_games)
        batch_texts = texts[start_idx:end_idx]
        batch_ids = game_ids[start_idx:end_idx]
        embeddings_texts = model.encode(batch_texts, convert_to_tensor=True, device=device)

        for theme, emb_theme in theme_embeddings.items():
            similarities = util.pytorch_cos_sim(embeddings_texts, emb_theme)
            sim_scaled = similarities.squeeze(dim=1) * 100.0 * (theme_weights[theme] / 5.0)
            sim_vals = sim_scaled.detach().cpu().numpy() if hasattr(sim_scaled, "detach") else np.array(sim_scaled)

            for i, game_id in enumerate(batch_ids):
                if game_id not in all_scores:
                    all_scores[game_id] = {}
                score_val = float(sim_vals[i]) if np.isfinite(sim_vals[i]) else 0.0
                score_val = max(0.0, min(score_val, 100.0))
                all_scores[game_id][theme] = round(score_val, 2)

        for i, game_id in enumerate(batch_ids):
            try:
                game_title = next(g["title"] for g in games if str(g["app_id"]) == str(game_id))
            except StopIteration:
                game_title = game_id
            print(f"[Embedding Evaluation] {start_idx + i + 1}/{num_games}: {game_title} → {all_scores.get(game_id, {})}", flush=True)

    for game_id, scores in all_scores.items():
        theme_values = list(scores.values())
        all_scores[game_id]["Embedding_Total"] = round(float(np.mean(theme_values)), 2) if theme_values else 0.0

    build_tables(games, all_scores, "embedding_evaluation")
