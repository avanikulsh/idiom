"""
Load and process English idiom corpus.
"""
import json
import csv
from pathlib import Path
from typing import List, Dict
import re


class IdiomLoader:
    """Load English idioms from various file formats."""

    @staticmethod
    def load_from_txt(file_path: Path, delimiter: str = '\n') -> List[str]:
        """
        Load idioms from a plain text file.

        Args:
            file_path: Path to the text file
            delimiter: Delimiter between idioms (default: newline)

        Returns:
            List of idioms
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        idioms = [idiom.strip() for idiom in content.split(delimiter) if idiom.strip()]
        return idioms

    @staticmethod
    def load_from_json(file_path: Path, idiom_key: str = 'idiom') -> List[Dict]:
        """
        Load idioms from a JSON file.

        Args:
            file_path: Path to the JSON file
            idiom_key: Key in JSON object containing the idiom text

        Returns:
            List of idiom dictionaries
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        else:
            raise ValueError("Unexpected JSON structure")

    @staticmethod
    def load_from_csv(file_path: Path, idiom_column: str = 'idiom') -> List[Dict]:
        """
        Load idioms from a CSV file.

        Args:
            file_path: Path to the CSV file
            idiom_column: Name of the column containing idiom text

        Returns:
            List of idiom dictionaries
        """
        idioms = []

        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                idioms.append(row)

        return idioms

    @staticmethod
    def normalize_idiom(idiom: str) -> str:
        """
        Normalize idiom text (lowercase, remove extra spaces, etc.).

        Args:
            idiom: Raw idiom text

        Returns:
            Normalized idiom
        """
        # Remove extra whitespace
        idiom = re.sub(r'\s+', ' ', idiom)

        # Remove leading/trailing whitespace
        idiom = idiom.strip()

        # Optionally lowercase (for comparison, but keep original for display)
        # idiom = idiom.lower()

        return idiom

    @staticmethod
    def load_idiom_corpus(directory: Path, include_contexts: bool = True) -> List[Dict]:
        """
        Load all idiom files from a directory.

        Args:
            directory: Path to directory containing idiom files
            include_contexts: Whether to load contextual examples

        Returns:
            List of idiom dictionaries with 'text' and 'source' fields
        """
        idioms = []

        # Process TXT files
        for txt_file in directory.glob("*.txt"):
            # Skip if it's a context file (handled separately)
            if 'context' in txt_file.name.lower():
                continue

            txt_idioms = IdiomLoader.load_from_txt(txt_file)
            for idiom in txt_idioms:
                idioms.append({
                    "text": IdiomLoader.normalize_idiom(idiom),
                    "source": txt_file.name
                })

        # Process JSON files
        for json_file in directory.glob("*.json"):
            json_idioms = IdiomLoader.load_from_json(json_file)
            for item in json_idioms:
                idiom_text = item.get('idiom') or item.get('text') or item.get('phrase')
                if idiom_text:
                    idiom_entry = {
                        "text": IdiomLoader.normalize_idiom(idiom_text),
                        "meaning": item.get('meaning', ''),
                        "source": json_file.name
                    }

                    # Handle MAGPIE format with contextual examples
                    if 'examples' in item and include_contexts:
                        idiom_entry['examples'] = item['examples']
                        idiom_entry['contexts'] = [ex.get('sentence', '') for ex in item['examples']]
                    elif 'example' in item:
                        idiom_entry['example'] = item.get('example', '')

                    idioms.append(idiom_entry)

        # Process CSV files
        for csv_file in directory.glob("*.csv"):
            csv_idioms = IdiomLoader.load_from_csv(csv_file)

            # Group by idiom if multiple examples exist (for context-based CSVs)
            idiom_dict = {}
            for item in csv_idioms:
                idiom_text = item.get('idiom') or item.get('text') or item.get('phrase')
                if idiom_text:
                    normalized = IdiomLoader.normalize_idiom(idiom_text)

                    if normalized not in idiom_dict:
                        idiom_dict[normalized] = {
                            "text": normalized,
                            "meaning": item.get('meaning', ''),
                            "source": csv_file.name,
                            "contexts": []
                        }

                    # Add context if available
                    if 'sentence' in item and include_contexts:
                        idiom_dict[normalized]['contexts'].append(item['sentence'])
                    elif 'example' in item:
                        idiom_dict[normalized]['example'] = item.get('example', '')

            idioms.extend(idiom_dict.values())

        print(f"Loaded {len(idioms)} idioms from {directory}")
        return idioms


# Example data sources for English idioms (to add to README)
IDIOM_DATA_SOURCES = """
Suggested English Idiom Data Sources:

1. **The Idiom Connection** - https://www.idiomconnection.com/
   - Free idiom database with meanings

2. **UsingEnglish.com Idiom Reference** - https://www.usingenglish.com/reference/idioms/
   - Comprehensive idiom list with categories

3. **Wiktionary Idioms** - https://en.wiktionary.org/wiki/Category:English_idioms
   - Open-source idiom database

4. **GitHub Datasets**:
   - Search for "english idioms dataset" or "idiom corpus"

5. **Academic Resources**:
   - VNC Tokens Dataset (Verb-Noun Combinations)
   - PIE Corpus (Potentially Idiomatic Expressions)
"""
