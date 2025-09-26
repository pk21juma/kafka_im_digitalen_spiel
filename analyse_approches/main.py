import json
import re
import os
import torch
import math
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import Dataset

# Weißt einen Score pro Kategorie anhand einer Gewichtung zu
# - 1 = 0%
# - 2 = 10%
# - 3 = 30%
# - 4 = 50%
# - 5 = 70%
# Weitere Treffer werden nur anteilig und normalisiert angerechnet
# Nicht final, wird eventuell durch ein lineareres Scoring ersetzt
def calculate_score(text, keywords):
    total_score = 0

    for kw, weight in keywords.items():
        pattern = rf"\b{re.escape(kw.lower())}\b"
        matches = re.findall(pattern, text.lower())
        num_matches = len(matches)

        if num_matches > 0:
            # Basiswert (einfach skaliert aus Gewicht 1–5 → 4–20 Punkte)
            base = weight * 4

            # Logarithmische Abflachung
            # log(1 + n) wächst langsamer als n
            score_kw = base * math.log(1 + num_matches)

            total_score += score_kw

    # Clipping, damit es pro Kategorie nicht über 100 Punkte geht
    return round(min(total_score, 100), 2)


# Simpler Ansatz für Phase 1: Suchen nach Stichwörtern, gewichtet
def phase1_analyze_games(games, kafka_keywords):
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
        for category, keywords in kafka_keywords.items():
            category_scores[category] = round(calculate_score(text, keywords), 2)

        # Gewichtungen für Kategorien (eventuell überflüssig)
        weights = {"Autobiographisch": 1.2, "Werke": 1.0, "Figuren": 1.5}
        weighted_sum = sum(category_scores[c]*weights.get(c,1) for c in category_scores)
        total_weight = sum(weights.get(c,1) for c in category_scores)
        category_scores["Phase1_Total"] = round(weighted_sum/total_weight,2)

        all_scores[game["app_id"]] = category_scores

        print(f"[Phase1] {idx}/{len(games)}: {game.get('title')} → {category_scores}")

    return all_scores

# Wordembedding Ansatz für Phase 2 mittels mpnet
# Score nicht linear, muss eventuell noch überarbeitet werden
# Batch-Size muss ertestet werden für die Hardware, CPU = 1
def phase2_analyze_games(games, kafka_themes, model, batch_size=16):
    # Texte vorbereiten
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

        full_text = game.get("description", "") + " " + " ".join(achievement_texts)
        texts.append(full_text)
        game_ids.append(game["app_id"])

    # Theme embedding mit Gewichtung
    theme_embeddings = {}
    theme_weights = {}
    for theme, definition in kafka_themes.items():
        if theme in ["kafkaesque", "absurd", "meritocracy", "civil service"]:
            weight = 5
        elif theme in ["alienated", "bureaucratic", "isolated"]:
            weight = 3
        else:
            weight = 1
        theme_embeddings[theme] = model.encode(definition, convert_to_tensor=True, device='cuda')
        theme_weights[theme] = weight

    # Batchweise Berechnung der Scores
    all_scores = {}
    num_games = len(games)

    for start_idx in range(0, num_games, batch_size):
        end_idx = min(start_idx + batch_size, num_games)
        batch_texts = texts[start_idx:end_idx]
        batch_ids = game_ids[start_idx:end_idx]

        # Alle Embeddings für den Batch
        embeddings_texts = model.encode(batch_texts, convert_to_tensor=True, device='cuda')

        # Similarities pro Thema
        for theme, emb_theme in theme_embeddings.items():
            similarities = util.pytorch_cos_sim(embeddings_texts, emb_theme)  # shape: (batch_size,1)
            sim_percent = (similarities.squeeze() * 100 * (theme_weights[theme]/5)).cpu().numpy()

            for i, game_id in enumerate(batch_ids):
                if game_id not in all_scores:
                    all_scores[game_id] = {}

                score_val = max(0.0, float(sim_percent[i]))  # Float-Konvertierung + keine negativen Werte
                all_scores[game_id][theme] = round(min(score_val, 100.0), 2)

        # Debug Print
        for i, game_id in enumerate(batch_ids):
            game_title = next(g["title"] for g in games if g["app_id"] == game_id)
            print(f"[Phase2] {start_idx + i + 1}/{num_games}: {game_title} → {all_scores[game_id]}", flush=True)

    # Gesamwert pro Spiel
    for game_id in all_scores:
        theme_values = list(all_scores[game_id].values())
        all_scores[game_id]["Phase2_Total"] = round(np.mean(theme_values), 2) if theme_values else 0.0

    return all_scores

# Phase 3 mittels Promting. Noch sehr experimentell. Grundlegende Idee:
# Wir promten unsere Texte gegen Textstellen von Krauss(2022), wo rezeptionsästethische
# Konzepte zu Kafka näher erläutert werden + ein Beispiel aus der Arbeit
# 
# Aktueller Status: Modell ist mit 2048 Tokens vermutlich zu klein. texte müssen SINNVOLL eingegrenzt
# werden, Aussagekraft ist noch sehr zweifelshaft. Alternative Modelle müssten getestet werden 
# (fraglich zwecks Hardware). Aktuelle Ausgabe erzeugt keine Zahlen, da Pipeline testweise auf
# max_new_tokens=10 begrenzt. -> rough und nur in Ansätzen verwendbar, wenn Pipeline wieder angepasst
def phase3_analyze_games(games, kafka_lit, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device="cuda", batch_size=8):    
    # Modell & Tokenizer laden
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Left-padding aufgrund der Hardware, auf Ausgabe achten und eventuell anpassen
    tokenizer.padding_side = 'left'
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Pipeline
    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        device=0 if device == "cuda" and torch.cuda.is_available() else -1,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        batch_size=batch_size
    )
    
    # Bisher sehr unerfolgreicher Promt, Anpassung des Scorings könnte helfen
    base_prompt = """Analyse: Wie stark zeigt dieses Spiel die beschriebenen Kafka-Konzepte?

Kafka-Konzept:
{}

Spiel:
{}

Bewerte mit einer Zahl zwischen 0 und 100, wobei:
- Niedrige Werte (0-29): Konzept kaum oder gar nicht erkennbar
- Mittlere Werte (30-49): Konzept teilweise oder moderat erkennbar  
- Hohe Werte (50-100): Konzept deutlich oder stark erkennbar

Deine Bewertung als Zahl:"""

    prompts_data = []
    
    # Konservativer, ingesamt 2048 möglich (inkl. Ausgabe)
    # Wird noch getestet, nicht final
    max_context_tokens = 1500
    
    for game in games:
        title = game.get("title", "Unknown")
        description = game.get("description", "")
        achievements = game.get("achievements", [])
        
        for section in kafka_lit["krauss_sections"]:
            section_id = section["id"]
            section_text = section["text"]
            
            # Hier berechnen wir unsere mögliche Tokenanzahl
            section_tokens = len(tokenizer.encode(section_text))
            base_prompt_tokens = len(tokenizer.encode(base_prompt.format("", "")))
            available_tokens = max_context_tokens - base_prompt_tokens - section_tokens - 150  # Mehr Buffer
            
            # texte vorbereiten
            game_text = prepare_game_text(description, achievements, tokenizer, available_tokens)
                
            prompt = base_prompt.format(section_text, game_text)
            
            # Debug Check, ob Tokensize eingehalten
            if len(tokenizer.encode(prompt)) > max_context_tokens:
                print(f"[Phase3] Überspringe {title} - {section_id}: Prompt zu lang nach Vorbereitung")
                continue
            
            prompts_data.append({
                'prompt': prompt,
                'app_id': game["app_id"],
                'title': title,
                'section_id': section_id
            })
    
    print(f"[Phase3] Verarbeite {len(prompts_data)} Prompts in Batches von {batch_size}")
    
    all_results = []
    
    for i in range(0, len(prompts_data), batch_size):
        batch = prompts_data[i:i + batch_size]
        batch_prompts = [item['prompt'] for item in batch]
        
        try:
            # Output Pipeline, eventuell mergen mit initialer Pipeline
            outputs = pipe(
                batch_prompts,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                return_full_text=False,
                truncation=True,
                padding=True
            )
            
            # Zahl aus Output extrahieren
            for j, output in enumerate(outputs):
                original_data = batch[j]
                generated_text = output[0]["generated_text"] if isinstance(output, list) else output["generated_text"]
                
                # Verbesserte Score-Extraktion
                score = extract_score_from_output(generated_text, original_data['title'], original_data['section_id'])
                
                all_results.append({
                    'app_id': original_data['app_id'],
                    'title': original_data['title'],
                    'section_id': original_data['section_id'],
                    'score': score,
                    'raw_output': generated_text.strip()
                })
        
        except Exception as e:
            print(f"[Phase3] Fehler bei Batch {i//batch_size + 1}: {e}")
            # Fallback mit Fehlerwert
            for item in batch:
                all_results.append({
                    'app_id': item['app_id'],
                    'title': item['title'],
                    'section_id': item['section_id'],
                    'score': -1,
                    'raw_output': f"Batch-Fehler: {str(e)}"
                })
        
        print(f"[Phase3] Batch {i//batch_size + 1}/{(len(prompts_data)-1)//batch_size + 1} verarbeitet")
    
    # Ergebnisse (noch keine .csv Funktionalität)
    results_df = pd.DataFrame(all_results)
    
    final_results = []
    
    for app_id in results_df['app_id'].unique():
        game_data = results_df[results_df['app_id'] == app_id]
        title = game_data['title'].iloc[0]
        
        scores_dict = {'app_id': app_id, 'title': title}
        
        # Scores pro Sektion
        section_scores = []
        for _, row in game_data.iterrows():
            if row['score'] >= 0:  # Nur gültige Scores
                scores_dict[row['section_id']] = row['score']
                section_scores.append(row['score'])
            else:
                # Explizit als fehlend markieren!!!
                scores_dict[row['section_id']] = None
        
        # Durchnitt nur mit den gültigen Scores
        if section_scores:
            scores_dict['krauss_total'] = round(sum(section_scores) / len(section_scores), 2)
            print(f"[Phase3] {title} → Durchschnitt: {scores_dict['krauss_total']} aus {len(section_scores)} gültigen Sektionen")
        else:
            scores_dict['krauss_total'] = None
            print(f"[Phase3] {title} → Keine gültigen Scores erhalten")
        
        final_results.append(scores_dict)
    
    return pd.DataFrame(final_results)

# Hilfsfunktion um die eigentlichen Scores in den Outputs zu erkenen und
# zu extrahieren, noch nicht final getestet
def extract_score_from_output(output_text, title, section_id):
    # Debug Print
    # print(f"[Debug] Raw output für {title}, {section_id}: '{output_text.strip()}'")
    
    # Bereinige den Output
    cleaned_output = output_text.strip()
    
    # Eventuell muss noch anders nach der Zahl gesucht werden
    # z.B. ohne Wortgrenzen
    
    # Alle 1-3 stelligen Zahlen durchsuchen
    all_numbers = re.findall(r'\b(\d{1,3})\b', cleaned_output)
    for num_str in all_numbers:
        score = int(num_str)
        if 0 <= score <= 100:
            print(f"[Debug] Gültige Zahl gefunden: {score}")
            return score
    
    #loose_numbers = re.findall(r'(\d{1,3})', cleaned_output)
    #for num_str in loose_numbers:
    #    score = int(num_str)
    #    if 0 <= score <= 100:
    #        print(f"[Debug] Gültige Zahl gefunden: {score}")
    #        return score
    
    print(f"[Phase3] Keine gültige Zahl für {title}, {section_id}: '{cleaned_output}'")
    return -1

# Da wir eine strikte Begrenzung in der Tokenzahl haben, müssen wir
# unsere gescrapten Texte eventuell begrenzen.
# Dafür haben wir zwei Möglichkeiten: Achievements, oder Beschreibungen
# priorisieren
def prepare_game_text(description, achievements, tokenizer, max_tokens):
    
    # Achievements sammeln und formatieren
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
    
    # Priorität über die Achievements
    if achievements_parts:
        achievements_text = " | ".join(achievements_parts)
        achievements_tokens = len(tokenizer.encode(achievements_text))
        
        if achievements_tokens <= max_tokens:
            # Prüfe ob noch Platz für Beschreibung ist
            if description.strip():
                remaining_tokens = max_tokens - achievements_tokens

                if remaining_tokens > 50:
                    truncated_desc = truncate_text(description, tokenizer, remaining_tokens)

                    if truncated_desc.strip():
                        return f"{achievements_text} | {truncated_desc}"
                    
            return achievements_text
        
        else:
            # Achievements kürzen
            return truncate_text(achievements_text, tokenizer, max_tokens)
    
    # Fallback: Nur Beschreibung
    if description.strip():
        return truncate_text(description, tokenizer, max_tokens)
    
    return ""

# Sehr rudimentäre Implementation zur Kürzung der gescrapten Texte
# Finale Version könnte anders aussehen
def truncate_text(text, tokenizer, max_tokens):
    if not text.strip():
        return ""
    
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text
    
    truncated_tokens = tokens[:max_tokens]
    truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    
    # Wortgrenze
    words = truncated_text.split()
    if len(words) > 1:
        # Abgeschnittene Wörter entfernen
        return " ".join(words[:-1])
    
    return truncated_text

def build_tables(games, scores_dict, phase_name, top10_dir="top10", out_csv="supertable.csv"):
    rows = []
    for game in games:
        row = {
            "game_id": game.get("app_id"),
            "title": game.get("title"),
            "steam_link": f"https://store.steampowered.com/app/{game.get('app_id')}"
        }
        game_scores = scores_dict.get(game["app_id"], {})
        row.update(game_scores)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"{phase_name} Tabelle gespeichert: {out_csv}")

    # Top-10 pro Kategorie
    top10_phase_dir = os.path.join(top10_dir, phase_name.lower())
    os.makedirs(top10_phase_dir, exist_ok=True)
    for category in df.columns:
        if category in ["game_id", "title", "steam_link"]:
            continue
        top10 = df.sort_values(by=category, ascending=False).head(10)
        top10_path = os.path.join(top10_phase_dir, f"top10_{category}.csv")
        top10.to_csv(top10_path, index=False, encoding="utf-8")
        print(f"   → Top-10 '{category}' gespeichert in {top10_path}")

    return df

# ---- Main ----
def main():
    # Daten laden
    with open("filtered_games.json", "r", encoding="utf-8") as f:
        games = json.load(f)

    with open("kafka_keywords.json", "r", encoding="utf-8") as f:
        kafka_keywords = json.load(f)

    with open("kafka_themes.json", "r", encoding="utf-8") as f:
        kafka_themes = json.load(f)

    with open("kafka_literature.json", "r", encoding="utf-8") as f:
        kafka_lit = json.load(f)

    # Modell laden für Phase 2
    #model = SentenceTransformer('all-mpnet-base-v2')

    scores_phase1 = phase1_analyze_games(games, kafka_keywords)
    df_phase1 = build_tables(games, scores_phase1, phase_name="Phase1")

    # Für optimale Laufzeiten, sollte das Modell auf der GPU ausgeführt werden, hierzu ist
    # die richtige Torch Version nötig, siehe: https://pytorch.org/get-started/locally/
    # für neuere Hardware reicht eine einfache Torch Installation über den pip Command
    #print(torch.cuda.is_available())
    #scores_phase2 = phase2_analyze_games(games, kafka_themes, model)
    #df_phase2 = build_tables(games, scores_phase2, phase_name="Phase2")

    #df = phase3_analyze_games(games[:10], kafka_lit)
    # print(df)

if __name__ == "__main__":
    main()
