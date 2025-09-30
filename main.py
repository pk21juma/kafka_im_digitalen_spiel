import scraper
import korpus
import analyse_approches
import json

def main():
    #scraper.scrape_by_tags(korpus.tags_to_scrape)
    #scraper.scrape_by_filter(
    #    input_file=r"korpus\games.json",
    #    output_file=r"korpus\filtered_games.json",
    #    progress_file=r"scraper\filter_progress.json"
    #)

    with open(r"korpus/filtered_games.json", "r", encoding="utf-8") as f:
        games = json.load(f)
    
    with open(r"korpus/kafka_keywords.json", "r", encoding="utf-8") as f:
        kafka_keywords = json.load(f)

    # analyse_approches.keyword_evaluation(games, kafka_keywords)

    with open(r"korpus/kafka_themes.json", "r", encoding="utf-8") as f:
        kafka_themes = json.load(f)

    # analyse_approches.embedding_evaluation(games, kafka_themes)

    # analyse_approches.create_combined_baseline_score()

    with open(r"korpus/kafka_literature.json", "r", encoding="utf-8") as f:
        kafka_literature = json.load(f)

    analyse_approches.generative_evaluation(games, kafka_literature, batch_size=8, frame = 10)

    # analyse_approches.generative_evaluation(games, kafka_literature, generator_model="microsoft/phi-2", batch_size=2, frame = 10)
if __name__ == "__main__":
    main()
