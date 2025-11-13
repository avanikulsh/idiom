"""
Configuration file for the idiom extraction and similarity project.
"""
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"

# Raw data paths
ENGLISH_IDIOMS_DIR = RAW_DATA_DIR / "english_idioms"
SUBTITLES_DIR = RAW_DATA_DIR / "subtitles"

# Subtitle language directories
SPANISH_SUBTITLES = SUBTITLES_DIR / "spanish"
HINDI_SUBTITLES = SUBTITLES_DIR / "hindi"
OTHER_SUBTITLES = SUBTITLES_DIR / "other"

# Model settings
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# Alternative models:
# - "sentence-transformers/LaBSE"
# - "sentence-transformers/distiluse-base-multilingual-cased-v2"
# - "xlm-roberta-base"

# MWE extraction settings
MIN_MWE_LENGTH = 2  # Minimum number of words in an MWE
MAX_MWE_LENGTH = 8  # Maximum number of words in an MWE
MIN_FREQUENCY = 2   # Minimum frequency for an MWE to be considered

# Similarity thresholds
SIMILARITY_THRESHOLD = 0.6  # Cosine similarity threshold for matching

# Languages configuration
LANGUAGES = {
    "spanish": {
        "code": "es",
        "spacy_model": "es_core_news_sm",
        "dir": SPANISH_SUBTITLES
    },
    "hindi": {
        "code": "hi",
        "spacy_model": "xx_ent_wiki_sm",  # Multilingual model
        "dir": HINDI_SUBTITLES
    }
}
