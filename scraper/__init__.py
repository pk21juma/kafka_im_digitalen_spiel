"""
scraper.py

Dieses Modul enthält Funktionalitäten zum Scrapen von Achievements und Beschreibungstexten auf 
Basis der Plattform Steam. Spieletitel können nach Tags gescraped werden.
"""

from .scrape_by_tags import scrape_by_tags
from .scrape_by_filter import scrape_by_filter