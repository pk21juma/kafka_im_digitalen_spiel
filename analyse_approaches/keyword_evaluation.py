from .scoring import calculate_score
from .utils import build_tables

def keyword_evaluation(games, keywords):
    """
    Evaluates the description and achievements by explicit references of
    a given games list and a set of weighted keywords.
    """
    all_scores = {}

    for idx, game in enumerate(games, 1):
        achievement_texts = []
        for ach in game.get("achievements", []):
            if isinstance(ach, dict):
                achievement_texts.append(ach.get("title", ""))
                achievement_texts.append(ach.get("description", ""))
            elif isinstance(ach, str):
                achievement_texts.append(ach)

        text = " ".join(achievement_texts).lower()
        category_scores = {}
        for category, keyword in keywords.items():
            category_scores[category] = round(calculate_score(text, keyword), 2)

        weights = {"Autobiographisch": 1.2, "Werke": 1.0, "Figuren": 1.5}
        weighted_sum = sum(category_scores[c] * weights.get(c, 1) for c in category_scores)
        total_weight = sum(weights.get(c, 1) for c in category_scores)
        category_scores["K_E_Total"] = round(weighted_sum / total_weight, 2)

        all_scores[game["app_id"]] = category_scores
        print(f"[Keyword evaluation] {idx}/{len(games)}: {game.get('title')} → {category_scores}")

    build_tables(games, all_scores, "keyword_evaluation")