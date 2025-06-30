import os
import logging
from tavily import TavilyClient
from deep_translator import GoogleTranslator
from typing import List, Dict

logger = logging.getLogger(__name__)

class TavilySearchAgent:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("❌ Clé API Tavily requise")
        self.client = TavilyClient(api_key=api_key)
        self.translator = GoogleTranslator(source='fr', target='en')

        # Mapping de termes économiques pour enrichir la requête
        self.economic_translations = {
            "croissance": "economic growth GDP",
            "inflation": "inflation rate CPI",
            "chômage": "unemployment rate",
            "dette": "public debt government debt",
            "pib": "GDP gross domestic product",
        }

    def translate_and_enhance_query(self, query: str) -> str:
        """Traduit et enrichit la requête en anglais"""
        enhanced_query = query.lower()
        # Remplacer des termes spécifiques
        for fr, en in self.economic_translations.items():
            if fr in enhanced_query:
                enhanced_query += f" {en}"
        # Traduire le reste
        translated = self.translator.translate(enhanced_query)
        logger.info(f"✅ Requête traduite et enrichie : {translated}")
        return translated

    def search(self, query: str, max_results: int = 5) -> List[Dict]:
            """Recherche Tavily avec requête traduite et enrichie"""
            enhanced_query = self.translate_and_enhance_query(query)
            response_dict = self.client.search( # Renommé en response_dict pour plus de clarté
                query=enhanced_query,
                search_depth="advanced",
                max_results=max_results,
                include_domains=["oecd.org", "imf.org", "worldbank.org", "ecb.europa.eu"]
            )
            # On extrait la liste de la clé "results".
            # .get("results", []) est une méthode sûre qui renvoie une liste vide si la clé n'existe pas.
            return response_dict.get("results", []) # <-- RETOURNE LA LISTE CONTENUE DANS LE DICTIONNAIRE
