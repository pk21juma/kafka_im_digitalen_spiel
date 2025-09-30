import pandas as pd
import numpy as np
import os
from .scoring import rescale_scores

def build_tables(
    games,
    scores_dict,
    evaluation_name,
    top10_dir=r"results/top10",
    out_csv=r"results/supertable.csv",
):
    """
    Builds a super tables for the calculated scores. Every approach can append.
    Furthermore top 10 scores tables a build per method and category.

    Args:
        games (list): game data.
        scores_dict (dict): {app_id: {category: score, ...}}
        evaluation_name (str): Name of approach (e.g: keyword, embedding, generative).
        top10_dir (str): Folder for top 10 tables.
        out_csv (str): Shared super table.
    """

    rows = []

    for game in games:
        row = {
            "game_id": str(game.get("app_id")),
            "title": game.get("title"),
            "steam_link": f"https://store.steampowered.com/app/{game.get('app_id')}",
        }

        game_scores = scores_dict.get(game["app_id"], {})
        row.update({f"{evaluation_name}_{k}": v for k, v in game_scores.items()})
        rows.append(row)

    df_new = pd.DataFrame(rows)

    if os.path.exists(out_csv):
        df_existing = pd.read_csv(out_csv, dtype={"game_id": str})
        df_merged = pd.merge(df_existing, df_new, on=["game_id", "title", "steam_link"], how="outer")
    else:
        df_merged = df_new

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_merged.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[Build Tables] {evaluation_name}: @ {out_csv} appended.")

    if top10_dir != "":
        top10_eval_dir = os.path.join(top10_dir, evaluation_name.lower())
        os.makedirs(top10_eval_dir, exist_ok=True)
        for category in df_new.columns:
            if category in ["game_id", "title", "steam_link"]:
                continue
            top10 = df_new.sort_values(by=category, ascending=False).head(10)
            top10_path = os.path.join(top10_eval_dir, f"top10_{category}.csv")
            top10.to_csv(top10_path, index=False, encoding="utf-8")
            print(f"[Build Tables] Top-10 '{category}' saved in {top10_path}.")

def truncate_text(text, tokenizer, max_tokens):
    """
    Shortens a given list of tokens/text.

    Args:
        text (str): Text to be shortened.
        tokenizer (): Tokenize Modell.
        max_tokens (int): Desired token count for return text.

    Returns:
        truncated_text (str): Text shortened to desired token length.
    """
    if not text.strip():
        return ""
    
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text
    
    truncated_tokens = tokens[:max_tokens]
    truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    
    words = truncated_text.split()
    if len(words) > 1:
        return " ".join(words[:-1])
    
    return truncated_text

def prepare_game_text(description, achievements, tokenizer, max_tokens):
    """
    Shortens the game data prioritising the achievements and filling the 
    rest with description text.

    Args:
        description (str): Text to be shortened.
        achievements (list): Text to be shortened.
        tokenizer (): Tokenize Modell.
        max_tokens (int): Desired token count for return text.

    Returns:
        truncated_text (): Prepared game data.
    """

    achievements_parts = []
    for ach in achievements:
        if isinstance(ach, dict):
            title = ach.get("title", "").strip()
            desc = ach.get("description", "").strip()
            if title and desc:
                achievements_parts.append(f"{title}: {desc}")
            elif title:
                achievements_parts.append(title)
            elif desc:
                achievements_parts.append(desc)
        elif isinstance(ach, str) and ach.strip():
            achievements_parts.append(ach.strip())
    
    if achievements_parts:
        achievements_text = " | ".join(achievements_parts)
        achievements_tokens = len(tokenizer.encode(achievements_text))
        
        if achievements_tokens <= max_tokens:
            if description.strip():
                remaining_tokens = max_tokens - achievements_tokens

                if remaining_tokens > 50:
                    truncated_desc = truncate_text(description, tokenizer, remaining_tokens)

                    if truncated_desc.strip():
                        return f"{achievements_text} | {truncated_desc}"
                    
            return achievements_text
        
        else:
            return truncate_text(achievements_text, tokenizer, max_tokens)
    
    if description.strip():
        return truncate_text(description, tokenizer, max_tokens)
    
    return ""

def create_combined_baseline_score(
    supertable_path=r"results/supertable.csv",
    method="zscore"
):
    """
    Creates a combined baseline score row from both the total score
    of embedding evalution and keyword evaluation.
    
    Args:
        supertable_path (str): Path to supertable.csv.
        method (str): Rescaling method. (Either zscore or min-max)
        output_path (str): Optionaler Output-Pfad (default: überschreibt Input)
    """
    
    output_path = supertable_path
    
    df = pd.read_csv(supertable_path, dtype={"game_id": str})

    keyword_total_col = "keyword_evaluation_K_E_Total"
    embedding_total_col = "embedding_evaluation_Embedding_Total"

    keyword_valid = df[keyword_total_col].notna()
    n_keyword_valid = keyword_valid.sum()
    print(f"\n[Rescale Combined] Keyword Total: {n_keyword_valid} valid scores.")
    
    if n_keyword_valid > 0:
        keyword_scores = df.loc[keyword_valid, keyword_total_col].values
        keyword_rescaled = rescale_scores(keyword_scores, method=method)
        df.loc[keyword_valid, 'keyword_rescaled'] = keyword_rescaled
        df.loc[~keyword_valid, 'keyword_rescaled'] = np.nan
        print(f"Rescaled: min={keyword_rescaled.min():.2f}, max={keyword_rescaled.max():.2f}")
    else:
        df['keyword_rescaled'] = np.nan
    
    embedding_valid = df[embedding_total_col].notna()
    n_embedding_valid = embedding_valid.sum()
    print(f"\n[Rescale Combined] Embedding Total: {n_embedding_valid} valid scores.")
    
    if n_embedding_valid > 0:
        embedding_scores = df.loc[embedding_valid, embedding_total_col].values
        embedding_rescaled = rescale_scores(embedding_scores, method=method)
        df.loc[embedding_valid, 'embedding_rescaled'] = embedding_rescaled
        df.loc[~embedding_valid, 'embedding_rescaled'] = np.nan
        print(f"  Rescaled: min={embedding_rescaled.min():.2f}, max={embedding_rescaled.max():.2f}")
    else:
        df['embedding_rescaled'] = np.nan
    
    df['baseline_combined_rescaled'] = df[['keyword_rescaled', 'embedding_rescaled']].mean(axis=1, skipna=True)
    
    valid_combined = df['baseline_combined_rescaled'].notna()
    n_valid = valid_combined.sum()
    print(f"\n[Rescale Combined] {n_valid}/{len(df)} games have valid scores.")
    
    if n_valid > 0:
        print(f"\n[Rescale Combined] Statistics:")
        print(f"  Min:  {df['baseline_combined_rescaled'].min():.2f}")
        print(f"  Max:  {df['baseline_combined_rescaled'].max():.2f}")
        print(f"  Mean: {df['baseline_combined_rescaled'].mean():.2f}")
        print(f"  Std:  {df['baseline_combined_rescaled'].std():.2f}")
    
    df['baseline_combined_rescaled'] = df['baseline_combined_rescaled'].round(2)
    df['keyword_rescaled'] = df['keyword_rescaled'].round(2)
    df['embedding_rescaled'] = df['embedding_rescaled'].round(2)
    
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n[Rescale Combined] Supertable aktualisiert: {output_path}")
    print(f"[Rescale Combined] Neue Spalte: 'baseline_combined_rescaled'")

    top10_dir = os.path.join(os.path.dirname(output_path), "top10", "combined_baseline")
    os.makedirs(top10_dir, exist_ok=True)
    
    top10 = df.sort_values(by='baseline_combined_rescaled', ascending=False).head(10)
    top10_path = os.path.join(top10_dir, "top10_baseline_combined.csv")
    top10.to_csv(top10_path, index=False, encoding="utf-8")
    print(f"[Rescale Combined] Top-10 gespeichert: {top10_path}")