import json
import os

def load_game_list(filename=r"korpus\games.json"):
    """
    Loads games list.

    Args:
        filename (str): Path to game list.

    Returns:
        list: Game list from JSON File.
        Array: Empty Array if Path does not exist.
    """
        
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_game_list(games, filename=r"korpus\games.json"):
    """
    Saves games list.

    Args:
        games (list): Game list to be saved.
        filename (str): Path for game list JSON file.
    """

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(games, f, ensure_ascii=False, indent=2)

def load_progress(filename=r"scraper\progress.json"):
    """
    Loads progress list.

    Args:
        filename (str): Path to progress file.

    Returns:
        list: Progress list from JSON File.
        list: New empty list if Path does not exist.
    """

    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed_tags": []}

def save_progress(progress, filename=r"scraper\progress.json"):
    """
    Saves progress list.

    Args:
        progress (list): Progress list to be saved.
        filename (str): Path for progress list JSON file.
    """
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)