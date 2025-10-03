from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import re
from .io_util import save_game_list, load_game_list, save_progress, load_progress

def scrape_by_single_tag(tag, all_games, existing_app_ids, progress,
                  games_filename=r"korpus\games.json", progress_filename=r"korpus\progress.json"):
    
    """
    Scrapes all Steam titles for a given tag.

    Args:
        tag (str): Tag to be scraped.
        all_games (list): List of all collected games.
        existing_app_ids (set): Already seen app_ids.
        progress (dict): Progress file.
        games_filename (str): Path to games list.
        progress_filename (str): Path to progress list.
    
    Returns:
        list: Updated list of all games.
    """

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

def scrape_by_tags(tags):
    """
    Calls scrape_by_single_tag iteratively over a give list of tags.

    Args:
        tags (list): List of Steam tags.
    """

    all_games = load_game_list()
    existing_app_ids = {game["app_id"] for game in all_games}
    progress = load_progress()

    for tag in tags:
        if tag in progress.get("completed_tags", []):
            print(f"Tag {tag} bereits abgeschlossen – überspringe")
            continue

        print(f"Scrape Tag: {tag}")
        all_games = scrape_by_single_tag(tag, all_games, existing_app_ids, progress)

        # Tag abgeschlossen, Offset löschen, speichern
        progress.setdefault("completed_tags", []).append(tag)
        progress["offsets"].pop(tag, None)
        save_progress(progress)
        save_game_list(all_games)

        print(f"Tag {tag} abgeschlossen, {len(all_games)} Spiele insgesamt")