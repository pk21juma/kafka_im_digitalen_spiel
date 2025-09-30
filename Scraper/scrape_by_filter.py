from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from bs4 import BeautifulSoup
import time
import re
import json
import os

def handle_agecheck(driver):
    """
    Takes a loaded Selenium Webdriver to perform an agecheck on
    the Steam shop site.

    Args:
        driver (Any): Loaded Selenium Webdriver.

    Returns:
        bool: Success of the agecheck operation.
    """

    try:
        # Entsprechende Dropdowns im Agecheck auswählen
        Select(driver.find_element(By.ID, "ageDay")).select_by_visible_text("1")
        Select(driver.find_element(By.ID, "ageMonth")).select_by_visible_text("January")
        Select(driver.find_element(By.ID, "ageYear")).select_by_visible_text("1980")

        # Und bestätigen
        btn = driver.find_element(By.ID, "view_product_page_btn")
        btn.click()
        time.sleep(1)
        return True
    
    except Exception as e:
        print(f"Agecheck nicht erfolgreich: {e}")
        return False


def get_game_details(driver, app_id: int, min_achievements=10, min_tokens=200):
    """
    Filters and srapes the description and achievemnts for a given steam title.

    Args:
        driver (Any): Loaded Selenium Webdriver.
        app_id (int): ID of a steam game title.
        min_achievements (int): Minimum number of achievements.
        min_tokens (int): Minimum number of tokens in game description.

    Returns:
        list: List with app_id, title, description and achievements.
    """
        
    url = f"https://store.steampowered.com/app/{app_id}/"
    driver.get(url)

    # Überprüfen, ob wir auf Agecheck URL umgeleitet wurden
    if "agecheck" in driver.current_url:
        # Überprüfen ob Agecheck erfolgreich war
        if not handle_agecheck(driver):
            return None

    # Warten auf das Laden des Hauptcontainers    
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "game_area_description"))
        )
    except:
        # Debug Print
        print(f"Timeout beim Laden: {app_id}")
        return None

    soup = BeautifulSoup(driver.page_source, "html.parser")

    # Überprüfen, ob Spiel ein DLC ist, Container dafür eindeutig
    # und im oberen Bereich der Seite
    if soup.find("div", class_="game_area_dlc_bubble"):
        print(f"{app_id} ist DLC, übersprungen.")
        return None

    # Überprüfen, ob Achievement Container vorhanden und, wenn
    # ob mindestens 10 Achievements enthalten sind
    achievements_block = soup.find("div", id="achievement_block")
    achievement_count = 0
    if achievements_block:
        match = re.search(r"Includes (\d+) Steam Achievements", achievements_block.text)
        if match:
            achievement_count = int(match.group(1))
    if achievement_count < min_achievements:
        print(f"{app_id} hat nur {achievement_count} Achievements, übersprungen.")
        return None

    # Speichern der Spielbeschreibung inklusive
    # Überprüfung, ob die Beschreibung eine Mindestanzahl an Tokens hat
    desc_elem = soup.find("div", id="game_area_description")
    description = desc_elem.get_text(separator=" ", strip=True) if desc_elem else ""
    token_count = len(description.split())
    if token_count < min_tokens:
        print(f"{app_id} hat nur {token_count} Tokens in Beschreibung, übersprungen.")
        return None

    # Titel vorsichtshalber erneut scrapen, falls dieser in der Listenansicht gekürzt wurde
    title_elem = soup.find("div", class_="apphub_AppName")
    title = title_elem.text.strip() if title_elem else "Unknown"

    # Achievements scrapen
    achievements = get_achievements(driver, app_id)

    return {
        "app_id": app_id,
        "title": title,
        #"review_count": review_count,
        "description": description,
        "achievements": achievements
    }


def get_achievements(driver, app_id):
    """
    Scrapes the achievemnts of a given steam title.

    Args:
        driver (Any): Loaded Selenium Webdriver.
        app_id (int): ID of a steam game title.

    Returns:
        Array: Achievements of the steam title.
    """
        
    url = f"https://steamcommunity.com/stats/{app_id}/achievements"
    driver.get(url)

    # Hauptcontainer ist hier trivial
    try:
        WebDriverWait(driver, 5).until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, "achieveRow"))
        )
    except:
        return []

    soup = BeautifulSoup(driver.page_source, "html.parser")
    rows = soup.find_all("div", class_="achieveRow")

    achievements = []
    for row in rows:
        title = row.find("h3").get_text(strip=True)
        desc = row.find("h5").get_text(strip=True)
        achievements.append({"title": title, "description": desc})

    return achievements

def scrape_by_filter(input_file=r"korpus\games.json", output_file=r"korpus\filtered_games.json", progress_file=r"korpus\filter_progress.json"):
    """
    Main Loop for the filter and scraping of all given steam titles in a list.
    Tracks progress until finished.

    Args:
        input_file (str): Initial games list with app_id.
        output_file (str): Filtered file appended with description and achievement list.
        progress_file (str): Tracks progress, gets removed after completion (maybe unstable).
    """
        
    with open(input_file, "r", encoding="utf-8") as f:
        games = json.load(f)

    # Wir speichern unsere aktuelle "Batch" unabhängig von der Hauptdatei
    # falls der Loop abbricht, können wir über diese Datei fortsetzen
    # Achtung, nicht vollumfänglich getestet, könnte Fehler produzieren
    if os.path.exists(progress_file):
        with open(progress_file, "r", encoding="utf-8") as f:
            progress = json.load(f)
        start_index = progress.get("last_index", 0)
        results = progress.get("results", [])
        # Debug Print
        print(f"Setze bei Index {start_index} fort. ({len(results)} Spiele gespeichert)")
    else:
        start_index = 0
        results = []

    total = len(games)

    # Driver Optionen, Sprache in Englisch aufgrund der Annahme, dass
    # jedes Spiel immer einen englischensprachigen Fallback hat
    options = webdriver.FirefoxOptions()
    options.set_preference("intl.accept_languages", "en-US, en")
    driver = webdriver.Firefox(options=options)

    # Hauptloop
    try:
        for i in range(start_index, total):
            game = games[i]
            app_id = game["app_id"]

            print(f"[{i+1}/{total}] Verarbeite {app_id}.")
            details = get_game_details(driver, app_id)
            if details:
                results.append(details)

            # Fortschritt speichern
            with open(progress_file, "w", encoding="utf-8") as f:
                json.dump({"last_index": i+1, "results": results}, f, indent=2, ensure_ascii=False)

            # Nach einem "Batch" von 50 spielen, speichern wir diese auch in der
            # Hauptliste
            if (i + 1) % 50 == 0:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"Zwischenspeicherung. ({len(results)} Ergebnisse).")

            time.sleep(1)

    finally:
        driver.quit()

    # final speichern
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Fortschritt Datei löschen
    if os.path.exists(progress_file):
        os.remove(progress_file)

    print(f"{len(results)} gefilterte Spiele gespeichert in {output_file}.")