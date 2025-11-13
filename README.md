# Cross-Lingual Idiom Matcher

A research project for extracting Multi-Word Expressions (MWEs) from movie subtitles in multiple languages and finding semantic similarities with English idioms using multilingual embeddings.

## Overview

This project combines NLP techniques and cross-lingual semantic similarity to:
1. **Extract MWEs** from movie subtitles in Spanish, Hindi, and other languages
2. **Match them semantically** with English idioms using multilingual transformer models (mBERT, XLM-R, LaBSE)
3. **Visualize results** through an interactive Next.js web interface

## Project Structure

```
idiom-proj/
├── app/                          # Next.js app router (visualization)
│   ├── page.tsx                 # Home page with project overview
│   └── results/                 # Results visualization page
├── components/                   # React components
├── lib/                         # TypeScript utilities
├── python/                      # Python NLP pipeline
│   ├── config.py               # Project configuration
│   ├── mwe_extraction/         # MWE extraction modules
│   │   └── extractor.py       # MWE extraction logic
│   ├── similarity/             # Semantic similarity modules
│   │   └── semantic_matcher.py # Cross-lingual matching
│   ├── data_processing/        # Data loading and preprocessing
│   │   └── idiom_loader.py    # English idiom corpus loader
│   └── utils/                  # Utility functions
│       └── subtitle_parser.py  # Subtitle file parsing
├── notebooks/                   # Jupyter notebooks for research
│   ├── 01_data_exploration.ipynb
│   ├── 02_mwe_extraction.ipynb
│   └── 03_semantic_similarity.ipynb
├── data/                        # Data directory (gitignored)
│   ├── raw/                    # Raw data
│   │   ├── english_idioms/    # English idiom corpus
│   │   └── subtitles/         # Movie subtitle files
│   │       ├── spanish/
│   │       ├── hindi/
│   │       └── other/
│   ├── processed/              # Extracted MWEs
│   └── results/                # Matching results (JSON/CSV)
├── requirements.txt             # Python dependencies
├── package.json                # Node.js dependencies
└── README.md                   # This file
```

## Getting Started

### Prerequisites

- **Python 3.9+** (for NLP processing)
- **Node.js 18+** (for Next.js visualization)
- **Git** (for version control)

### 1. Clone and Setup

```bash
cd idiom-proj
```

### 2. Python Environment Setup

Create a virtual environment and install dependencies:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Download spaCy Language Models

```bash
# English model
python -m spacy download en_core_web_sm

# Spanish model
python -m spacy download es_core_news_sm

# Multilingual model (for Hindi and others)
python -m spacy download xx_ent_wiki_sm
```

### 4. Next.js Setup

```bash
# Install Node dependencies
npm install

# Run development server
npm run dev
```

Visit [http://localhost:3000](http://localhost:3000) to see the web interface.

## Data Acquisition

### English Idioms

Add English idiom files to `data/raw/english_idioms/`. Supported formats:

- **TXT**: One idiom per line
- **CSV**: With columns like `idiom`, `meaning`, `example`
- **JSON**: Array of idiom objects

**Recommended data sources:**
- [The Idiom Connection](https://www.idiomconnection.com/)
- [UsingEnglish.com](https://www.usingenglish.com/reference/idioms/)
- [Wiktionary English Idioms](https://en.wiktionary.org/wiki/Category:English_idioms)
- Academic corpora: VNC Tokens, PIE Corpus

### Movie Subtitles

Add subtitle files (`.srt` or `.vtt` format) to:
- `data/raw/subtitles/spanish/` for Spanish
- `data/raw/subtitles/hindi/` for Hindi
- `data/raw/subtitles/other/` for other languages

**Where to find subtitles:**
- [OpenSubtitles](https://www.opensubtitles.org/)
- [Subscene](https://subscene.com/)
- Personal movie collections

## Usage Workflow

### Step 1: Data Exploration

Open and run `notebooks/01_data_exploration.ipynb`:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

This notebook:
- Loads English idiom corpus
- Loads subtitle files
- Analyzes data statistics
- Visualizes idiom length distributions

### Step 2: MWE Extraction

Run `notebooks/02_mwe_extraction.ipynb`:

```bash
jupyter notebook notebooks/02_mwe_extraction.ipynb
```

This notebook:
- Processes subtitle data for a selected language
- Extracts candidate MWEs using:
  - N-gram frequency analysis
  - spaCy noun phrase detection
  - Verb phrase patterns
- Filters and saves results to `data/processed/`

### Step 3: Semantic Similarity Matching

Run `notebooks/03_semantic_similarity.ipynb`:

```bash
jupyter notebook notebooks/03_semantic_similarity.ipynb
```

This notebook:
- Loads English idioms and foreign MWEs
- Encodes them using multilingual sentence transformers
- Computes cosine similarity scores
- Finds top-k matches for each English idiom
- Saves results to `data/results/` as JSON and CSV

### Step 4: Visualize Results

View results in the Next.js interface:

```bash
npm run dev
```

Navigate to `/results` to explore matches (requires implementing API route).

## Configuration

Edit `python/config.py` to customize:

```python
# Embedding model
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# MWE extraction parameters
MIN_MWE_LENGTH = 2
MAX_MWE_LENGTH = 8
MIN_FREQUENCY = 2

# Similarity threshold
SIMILARITY_THRESHOLD = 0.6
```

### Available Multilingual Models

- `paraphrase-multilingual-mpnet-base-v2` (recommended)
- `LaBSE` (Language-agnostic BERT Sentence Encoder)
- `distiluse-base-multilingual-cased-v2`
- `xlm-roberta-base`

## Python Modules

### `python/mwe_extraction/extractor.py`

```python
from python.mwe_extraction.extractor import MWEExtractor

extractor = MWEExtractor(language='es', spacy_model='es_core_news_sm')
mwes = extractor.extract_candidate_mwes(texts, min_length=2, max_length=6)
```

### `python/similarity/semantic_matcher.py`

```python
from python.similarity.semantic_matcher import SemanticMatcher

matcher = SemanticMatcher(model_name='paraphrase-multilingual-mpnet-base-v2')
matches = matcher.find_similar_mwes(english_idioms, foreign_mwes, threshold=0.6)
```

### `python/utils/subtitle_parser.py`

```python
from python.utils.subtitle_parser import load_subtitles_from_directory

subtitles = load_subtitles_from_directory(SPANISH_SUBTITLES)
```

## Next.js Development

### Run Development Server

```bash
npm run dev
```

### Build for Production

```bash
npm run build
npm start
```

### Adding API Routes

Create `app/api/results/route.ts` to load JSON results:

```typescript
import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

export async function GET() {
  const resultsPath = path.join(process.cwd(), 'data/results/idiom_mwe_matches_spanish.json');
  const data = JSON.parse(fs.readFileSync(resultsPath, 'utf-8'));
  return NextResponse.json(data);
}
```

## Research Notes

### MWE Extraction Techniques

1. **Frequency-based n-grams**: Extract recurring sequences
2. **Syntactic patterns**: Use spaCy's dependency parsing for noun/verb phrases
3. **Statistical measures**: PMI, t-test for collocation detection (future work)

### Semantic Similarity Approaches

- **Embeddings**: Use sentence transformers to encode idioms and MWEs
- **Cosine similarity**: Measure semantic closeness in embedding space
- **Cross-lingual models**: mBERT, XLM-R trained on parallel corpora

### Challenges

- **Idiomatic expressions** are often culturally specific
- **Literal translations** may not capture semantic equivalence
- **Subtitle quality** varies (OCR errors, informal language)
- **Context dependency**: MWEs may have different meanings in context

## Future Enhancements

- [ ] Add more languages (French, Arabic, Japanese)
- [ ] Implement context-aware matching using full sentence embeddings
- [ ] Add manual annotation interface for validation
- [ ] Integrate with translation APIs for additional features
- [ ] Build API for programmatic access
- [ ] Add clustering to find groups of similar idioms
- [ ] Implement active learning for model improvement

## Contributing

This is a research project. Contributions are welcome:

1. Add new data sources
2. Improve MWE extraction algorithms
3. Test different embedding models
4. Enhance visualization interface

## License

MIT License - feel free to use for research and education.

## Citation

If you use this project in your research, please cite:

```
@software{cross_lingual_idiom_matcher,
  title = {Cross-Lingual Idiom Matcher},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/idiom-proj}
}
```

## References

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [spaCy Language Processing](https://spacy.io/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- Research papers on MWE extraction and cross-lingual semantics

## Contact

For questions or collaboration, please open an issue on GitHub.
