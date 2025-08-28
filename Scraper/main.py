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

# Loop für das Scrapen der Spiele in der Listenansicht nach Tag
# Achtung, sehr "gehardcoded", da die Liste im Backend gerendert
# und die meistens Divs gehashed angezeigt werden
# Muss eventuell geupdatet werden, wenn die Struktur der Seite sich ändert
def scrape_by_tag(tag, all_games, existing_app_ids, progress,
                  games_filename="games.json", progress_filename="progress.json"):

    # Rudimentärer Speicheransatz, wir starten immer beim zuletzt
    # besuchten Hashtag und dem gespeicherten Offset
    offset = progress.get("offsets", {}).get(tag, 0)

    # Driver Optionen, Sprache in Englisch aufgrund der Annahme, dass
    # jedes Spiel immer einen englischensprachigen Fallback hat
    options = webdriver.FirefoxOptions()
    options.set_preference("intl.accept_languages", "en-US, en")
    driver = webdriver.Firefox(options=options)

    try:
        # keine Bedingung, da wir nicht verlässlich wissen, wie viele Spiele
        # unter einem Tag zu finden sind, Angaben von Steam unzuverlässlig
        while True:
            # der Offset ist unsere Seitenangabe, ein Offset von 12 entspricht einer Seite
            base_url = f"https://store.steampowered.com/tags/en/{tag}/?offset={offset}"
            print(f"Lade Seite: {base_url}")
            driver.get(base_url)

            # Tags sind nicht sehr eindeutig definiert, wir müssen überprüfen, ob ein
            # Tag auf die category url umgeleitet wird
            current_url = driver.current_url
            if "category" in current_url:
                # Offset an die category-URL hängen
                if "offset=" in current_url:
                    url = re.sub(r'offset=\d+', f"offset={offset}", current_url)
                else:
                    sep = "&" if "?" in current_url else "?"
                    url = f"{current_url}{sep}offset={offset}"
                if url != current_url:
                    print(f"Redirect erkannt, verwende stattdessen {url}")
                    driver.get(url)

            # Warten bis der Hauptcontainer geladen wird, nicht sehr stabil, manchmal
            # dauert das Laden mehr als 5 Sekunden, (warum?)
            try:
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.sale_item_browser"))
                )
            except Exception as e:
                print(f"Wartezeit abgelaufen bei Offset {offset}: {e}")
                break
            
            # Speichern des Hauptcontainers
            # Wir scrollen außerdem zum Hauptcontainer und warten, damit alle
            # Kinder geladen werden
            element = driver.find_element(By.CSS_SELECTOR, "div.sale_item_browser")
            driver.execute_script("arguments[0].scrollIntoView(true);", element)
            time.sleep(2)

            soup = BeautifulSoup(driver.page_source, 'html.parser')
            main_container = soup.find('div', class_='sale_item_browser')
            if not main_container:
                print("Hauptcontainer nicht gefunden")
                break
            
            # Innerhalb unseres Hauptcontainers können wir alle Spiele
            # über den Container "ImpressionTrackedElement" identifizieren
            game_containers = main_container.select('div.ImpressionTrackedElement')
            if not game_containers:
                print("Keine weiteren Spiele gefunden")
                break
            
            # Scrapen aller relevanten Angaben inlusive Tracking der neu hinzugefügten
            # Spiele (wichtig, da mehrere Tags pro Spiel möglich)
            new_count = 0
            for container in game_containers:
                try:
                    app_link_elem = container.select_one('a[href*="/app/"]')
                    if not app_link_elem:
                        continue
                    app_link = app_link_elem['href']
                    app_id_match = re.search(r'/app/(\d+)', app_link)
                    if not app_id_match:
                        continue
                    app_id = app_id_match.group(1)

                    # Hier überprüfen wir Duplikate
                    if app_id in existing_app_ids:
                        continue
                    existing_app_ids.add(app_id)

                    title = app_link_elem.get_text(strip=True)
                    if not title:
                        img = app_link_elem.find('img', alt=True)
                        title = img['alt'].strip() if img else "N/A"

                    review_elem = container.select_one('a[href*="#app_reviews_hash"]')
                    if not review_elem:
                        continue

                    # Review Count aus Text extrahieren
                    review_text = review_elem.get_text(" ", strip=True)
                    m = re.search(r'([\d,\.]+)\s+User Reviews', review_text)
                    if not m:
                        continue
                    review_count = int(m.group(1).replace(',', '').replace('.', ''))

                    all_games.append({
                        "app_id": app_id,
                        "title": title,
                        "review_count": review_count
                    })
                    new_count += 1
                
                except Exception as e:
                    print(f"Fehler bei der Extraktion: {e}")
                    continue
            
            # Debug Print
            print(f"Seite {offset//12 + 1}: {new_count} neue Spiele")

            # Speichern nach jeder Seite, inklusive Checkpoint falls Loop unterbrochen wird
            save_game_list(all_games, games_filename)
            progress.setdefault("offsets", {})[tag] = offset + 12
            save_progress(progress, progress_filename)

            # Abbruchbedingung, falls keine Container auf der neuen
            # Seite mehr gefunden werden können
            if len(game_containers) < 1:
                print("Keine weiteren Spiele gefunden. Tag abgeschlossen.")
                break
            
            # Nächste Seite + kurze Wartezeit, damit Scraper nicht blockiert wird
            offset += 12
            time.sleep(1)

    finally:
        driver.quit()

    return all_games

# Diverse Hilfunktionen für Laden/Speichern der Listen

def load_game_list(filename="games.json"):
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_game_list(games, filename="games.json"):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(games, f, ensure_ascii=False, indent=2)

def load_progress(filename="progress.json"):
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed_tags": []}

def save_progress(progress, filename="progress.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

# Bei Umleitung auf die /agecheck/ Url müssen wir diese handlen
# Sehr hacky, alternative Lösungen könnten mittels Cookies, oder
# Steam Account umgesetzt werden
def handle_agecheck(driver):
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

# Funktion für das Scrapen der Beschreibungen und Achievements
# Gleichzeitig filtern wir hier DLCs aus, nach Mindestanzahl
# von Tokens in der Beschreibung und Mindestanzahl 
# Achievments von 10
def get_game_details(driver, app_id, min_achievements=10, min_tokens=200):
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

# Url für die Achievemnts laden und scrapen
def get_achievements(driver, app_id):
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

# Loop für das Scrapen der Spieldetails
def filter_games(input_file="games.json", output_file="filtered_games.json", progress_file="filter_progress.json"):
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

def main():
    # Alle interessanten Tags
    tags_to_scrape = [
        "Story Rich",
        "Choices Matter",
        "Mystery",
        "Multiple Endings",
        "Psychological Horror",
        "Management",
        "Drama",
        "Interactive Fiction",
        "Surreal",
        "Narration",
        "Lore-Rich",
        "Investigation",
        "Psychological",
        "Nonlinear",
        "Abstract",
        "Philosophical",
        "Dark Comedy",
        "Conspiracy",
        "Dynamic Narration",
        "Narrative",
        "God Game",
        "Faith",
        "Well-Written"]

    # Fortschritt für den ersten Scrape Loop
    all_games = load_game_list()
    existing_app_ids = {game["app_id"] for game in all_games}
    progress = load_progress()

    # Loop für die Tagliste, Überprüfung ob Tag schin abgespeichert
    for tag in tags_to_scrape:
        if tag in progress.get("completed_tags", []):
            print(f"Tag {tag} bereits abgeschlossen – überspringe")
            continue

        print(f"Scrape Tag: {tag}")
        all_games = scrape_by_tag(tag, all_games, existing_app_ids, progress)

        # Tag abgeschlossen, Offset löschen, speichern
        progress.setdefault("completed_tags", []).append(tag)
        progress["offsets"].pop(tag, None)
        save_progress(progress)
        save_game_list(all_games)

        print(f"Tag {tag} abgeschlossen, {len(all_games)} Spiele insgesamt")

    
    # filter_games(input_file="games.json", output_file="filtered_games.json", progress_file="filter_progress.json")
    
if __name__ == "__main__":
    main()
