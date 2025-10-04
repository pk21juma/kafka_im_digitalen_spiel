import re
import os
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSequenceClassification
import torch
from .utils import build_tables, prepare_game_text


def generative_evaluation(
    games,
    kafka_lit,
    generator_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # or "microsoft/phi-2"
    nli_model="cross-encoder/nli-deberta-v3-small",
    device="cuda",
    batch_size=4,
    frame=100,
):
    """
    Generative evaluation with NLI:
    1. Select Top-N and Flop-N games based on Combined Baseline Score.
    2. Generate qualitative answers with an LLM about Kafka literature sections.
    3. Analyze answers with an NLI model (entailment = higher score).
    4. Append results to the global supertable.
    5. Export Top10/Flop10 tables and a debug CSV with all responses.

    Args:
        games (list): List of all available games (dicts with "app_id", "title", "description", "achievements").
        kafka_lit (dict): Kafka literature with sections.
        generator_model (str): Model for text generation.
        nli_model (str): Model for NLI classification.
        device (str): "cuda" or "cpu".
        batch_size (int): Batch size for generator.
        frame (int): Number of best and worst games to consider.
    """

    print(f"[Generative Evaluation] Starting with {frame * 2} games from Combined Baseline")

    df_super = pd.read_csv(r"results/supertable.csv", dtype={"game_id": str})
    df_top = df_super.nlargest(frame, "baseline_combined_rescaled")
    df_flop = df_super.nsmallest(frame, "baseline_combined_rescaled")

    top_game_ids = set(df_top["game_id"].astype(str))
    flop_game_ids = set(df_flop["game_id"].astype(str))

    top_games = [g for g in games if str(g["app_id"]) in top_game_ids]
    flop_games = [g for g in games if str(g["app_id"]) in flop_game_ids]

    print(f"[Generative Evaluation] Found {len(top_games)} top games")
    print(f"[Generative Evaluation] Found {len(flop_games)} flop games")

    print(f"[Generative Evaluation] Loading generator model: {generator_model}")
    gen_tokenizer = AutoTokenizer.from_pretrained(generator_model)
    gen_tokenizer.padding_side = "left"

    gen_model = AutoModelForCausalLM.from_pretrained(
        generator_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    gen_pipe = pipeline(
        "text-generation",
        model=gen_model,
        tokenizer=gen_tokenizer,
        batch_size=batch_size,
    )

    print(f"[Generative Evaluation] Loading NLI model: {nli_model}")
    nli_tokenizer = AutoTokenizer.from_pretrained(nli_model)
    nli_classifier = AutoModelForSequenceClassification.from_pretrained(nli_model)

    if device == "cuda":
        nli_classifier = nli_classifier.to("cuda")

    nli_pipe = pipeline(
        "text-classification",
        model=nli_classifier,
        tokenizer=nli_tokenizer,
        device=0 if device == "cuda" else -1,
        top_k=None,  # return all labels
    )

    base_prompt = """Input: Does this game contain elements of the following concept? Answer in 1-2 sentences with your reasoning.

Concept: {section_text}

Game: {game_text}

Output:"""

    max_context_tokens = 1200

    print(f"[Generative Evaluation] Preparing prompts for top games...")
    prompts_data_top_games = []

    for game in top_games:
        title = game.get("title", "Unknown")
        description = game.get("description", "")
        achievements = game.get("achievements", [])

        for section in kafka_lit["krauss_sections"]:
            section_id = section["id"]
            section_text = section["text_en"]

            base_tokens = len(gen_tokenizer.encode(base_prompt.format(section_text="", game_text="")))
            section_tokens = len(gen_tokenizer.encode(section_text))
            available_tokens = max_context_tokens - base_tokens - section_tokens - 100

            game_text = prepare_game_text(description, achievements, gen_tokenizer, available_tokens)
            prompt = base_prompt.format(section_text=section_text, game_text=game_text)

            prompts_data_top_games.append({
                "prompt": prompt,
                "app_id": game["app_id"],
                "title": title,
                "section_id": section_id,
                "section_text": section_text,
            })

    print(f"[Generative Evaluation] {len(prompts_data_top_games)} prompts prepared for top games")

    top_games_results = []
    for i in range(0, len(prompts_data_top_games), batch_size):
        batch = prompts_data_top_games[i: i + batch_size]
        batch_prompts = [item["prompt"] for item in batch]

        try:
            outputs = gen_pipe(
                batch_prompts,
                max_new_tokens=80,
                do_sample=False,
                return_full_text=False,
            )

            for j, output in enumerate(outputs):
                original = batch[j]
                generated_text = (
                    output[0]["generated_text"] if isinstance(output, list) else output["generated_text"]
                )

                top_games_results.append({
                    "app_id": original["app_id"],
                    "title": original["title"],
                    "section_id": original["section_id"],
                    "section_text": original["section_text"],
                    "generated_answer": generated_text.strip(),
                })

                print(f"[Generative Evaluation] Output generated for {original['app_id']}")

        except Exception as e:
            print(f"[Generative Evaluation] Error in batch {i // batch_size + 1}: {e}")
            for item in batch:
                top_games_results.append({
                    "app_id": item["app_id"],
                    "title": item["title"],
                    "section_id": item["section_id"],
                    "section_text": item["section_text"],
                    "generated_answer": f"Error: {str(e)}",
                })

    print(f"[Generative Evaluation] Preparing prompts for flop games...")
    prompts_data_flop_games = []

    for game in flop_games:
        title = game.get("title", "Unknown")
        description = game.get("description", "")
        achievements = game.get("achievements", [])

        for section in kafka_lit["krauss_sections"]:
            section_id = section["id"]
            section_text = section["text_en"]

            base_tokens = len(gen_tokenizer.encode(base_prompt.format(section_text="", game_text="")))
            section_tokens = len(gen_tokenizer.encode(section_text))
            available_tokens = max_context_tokens - base_tokens - section_tokens - 100

            game_text = prepare_game_text(description, achievements, gen_tokenizer, available_tokens)
            prompt = base_prompt.format(section_text=section_text, game_text=game_text)

            prompts_data_flop_games.append({
                "prompt": prompt,
                "app_id": game["app_id"],
                "title": title,
                "section_id": section_id,
                "section_text": section_text,
            })

    print(f"[Generative Evaluation] {len(prompts_data_flop_games)} prompts prepared for flop games")

    flop_games_results = []
    for i in range(0, len(prompts_data_flop_games), batch_size):
        batch = prompts_data_flop_games[i: i + batch_size]
        batch_prompts = [item["prompt"] for item in batch]

        try:
            outputs = gen_pipe(
                batch_prompts,
                max_new_tokens=80,
                do_sample=False,
                return_full_text=False,
            )

            for j, output in enumerate(outputs):
                original = batch[j]
                generated_text = (
                    output[0]["generated_text"] if isinstance(output, list) else output["generated_text"]
                )

                flop_games_results.append({
                    "app_id": original["app_id"],
                    "title": original["title"],
                    "section_id": original["section_id"],
                    "section_text": original["section_text"],
                    "generated_answer": generated_text.strip(),
                })

                print(f"[Generative Evaluation] Output generated for {original['app_id']}")

        except Exception as e:
            print(f"[Generative Evaluation] Error in batch {i // batch_size + 1}: {e}")
            for item in batch:
                flop_games_results.append({
                    "app_id": item["app_id"],
                    "title": item["title"],
                    "section_id": item["section_id"],
                    "section_text": item["section_text"],
                    "generated_answer": f"Error: {str(e)}",
                })

    print(f"[Generative Evaluation: NLI] Analyzing top games...")
    for result in top_games_results:
        try:
            hypothesis = f"This game contains elements described as: {result['section_text'][:200]}"
            premise = result["generated_answer"]

            nli_result = nli_pipe(f"{premise} [SEP] {hypothesis}")

            entailment_score = 0.0
            for item in nli_result[0]:
                if item["label"].lower() in ["entailment", "entail"]:
                    entailment_score = item["score"] * 100
                    break

            result["nli_score"] = round(entailment_score, 2)
            result["nli_raw"] = nli_result[0]

        except Exception as e:
            print(f"[Generative Evaluation: NLI] Error for {result['title']}/{result['section_id']}: {e}")
            result["nli_score"] = None
            result["nli_raw"] = str(e)

    print(f"[Generative Evaluation: NLI] Analyzing flop games...")
    for result in flop_games_results:
        try:
            hypothesis = f"This game contains elements described as: {result['section_text'][:200]}"
            premise = result["generated_answer"]

            nli_result = nli_pipe(f"{premise} [SEP] {hypothesis}")

            entailment_score = 0.0
            for item in nli_result[0]:
                if item["label"].lower() in ["entailment", "entail"]:
                    entailment_score = item["score"] * 100
                    break

            result["nli_score"] = round(entailment_score, 2)
            result["nli_raw"] = nli_result[0]

        except Exception as e:
            print(f"[Generative Evaluation: NLI] Error for {result['title']}/{result['section_id']}: {e}")
            result["nli_score"] = None
            result["nli_raw"] = str(e)

    top_results_df = pd.DataFrame(top_games_results)
    flop_results_df = pd.DataFrame(flop_games_results)
    results_df = pd.concat([top_results_df, flop_results_df], ignore_index=True)

    all_scores = {}
    for app_id in results_df["app_id"].unique():
        game_data = results_df[results_df["app_id"] == app_id]
        scores = {}
        section_scores = [row["nli_score"] for _, row in game_data.iterrows() if row["nli_score"] is not None]

        for _, row in game_data.iterrows():
            scores[row["section_id"]] = row["nli_score"]

        if section_scores:
            scores["NLI_Total"] = round(np.mean(section_scores), 2)
            print(f"[Generative Evaluation: NLI] {app_id} → Mean: {scores['NLI_Total']}")
        else:
            scores["NLI_Total"] = None
            print(f"[Generative Evaluation: NLI] {app_id} → No valid scores")

        all_scores[app_id] = scores

    build_tables(top_games + flop_games, all_scores, "generative_evaluation", top10_dir="")

    df = pd.DataFrame([
        {
            "game_id": g.get("app_id"),
            "title": g.get("title"),
            "steam_link": f"https://store.steampowered.com/app/{g.get('app_id')}",
            **all_scores.get(str(g["app_id"]), {})
        }
        for g in games
    ])

    top10_gen_dir = os.path.join("results", "top10", "generative_evaluation")
    os.makedirs(top10_gen_dir, exist_ok=True)

    for category in df.columns:
        if category in ["game_id", "title", "steam_link"]:
            continue

        # Top-N
        top = df.sort_values(by=category, ascending=False).head(frame)
        top_path = os.path.join(top10_gen_dir, f"top_{frame}_{category}.csv")
        top.to_csv(top_path, index=False, encoding="utf-8")

        # Flop-N
        flop = df.sort_values(by=category, ascending=True).head(frame)
        flop_path = os.path.join(top10_gen_dir, f"flop_{frame}_{category}.csv")
        flop.to_csv(flop_path, index=False, encoding="utf-8")

        print(f"[Generative Evaluation] Top-{frame} '{category}' → {top_path}")
        print(f"[Generative Evaluation] Flop-{frame} '{category}' → {flop_path}")
    
    debug_path = "results/generative_debug_responses.csv"
    results_df.to_csv(debug_path, index=False, encoding="utf-8")
    print(f"\n[Generative Evaluation: NLI] Debug responses saved: {debug_path}")