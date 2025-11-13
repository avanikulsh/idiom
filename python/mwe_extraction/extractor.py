"""
Multi-Word Expression (MWE) extraction from text.
"""
import spacy
from typing import List, Dict, Tuple, Set
from collections import Counter
import re


class MWEExtractor:
    """Extract Multi-Word Expressions from text using various methods."""

    def __init__(self, language: str = "en", spacy_model: str = "en_core_web_sm"):
        """
        Initialize the MWE extractor.

        Args:
            language: Language code (e.g., 'en', 'es', 'hi')
            spacy_model: spaCy model to use
        """
        self.language = language
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"Model {spacy_model} not found. Download it with: python -m spacy download {spacy_model}")
            self.nlp = None

    def extract_ngrams(self, texts: List[str], n: int = 3, min_freq: int = 2) -> List[Tuple[str, int]]:
        """
        Extract n-grams from texts based on frequency.

        Args:
            texts: List of text strings
            n: N-gram size
            min_freq: Minimum frequency threshold

        Returns:
            List of (ngram, frequency) tuples, sorted by frequency
        """
        ngrams = []

        for text in texts:
            words = text.lower().split()
            for i in range(len(words) - n + 1):
                ngram = ' '.join(words[i:i+n])
                ngrams.append(ngram)

        # Count frequencies
        ngram_counts = Counter(ngrams)

        # Filter by minimum frequency
        filtered = [(ng, count) for ng, count in ngram_counts.items() if count >= min_freq]

        return sorted(filtered, key=lambda x: x[1], reverse=True)

    def extract_noun_phrases(self, texts: List[str]) -> List[str]:
        """
        Extract noun phrases using spaCy's parser.

        Args:
            texts: List of text strings

        Returns:
            List of noun phrases
        """
        if self.nlp is None:
            return []

        noun_phrases = []

        for text in texts:
            doc = self.nlp(text)
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) >= 2:  # Only multi-word
                    noun_phrases.append(chunk.text.lower())

        return list(set(noun_phrases))

    def extract_verb_phrases(self, texts: List[str]) -> List[str]:
        """
        Extract verb phrases (verb + object/complement patterns).

        Args:
            texts: List of text strings

        Returns:
            List of verb phrases
        """
        if self.nlp is None:
            return []

        verb_phrases = []

        for text in texts:
            doc = self.nlp(text)
            for token in doc:
                if token.pos_ == "VERB":
                    # Get verb and its children (objects, complements)
                    phrase_tokens = [token.text]
                    for child in token.children:
                        if child.dep_ in ["dobj", "pobj", "attr", "prep"]:
                            phrase_tokens.append(child.text)
                            # Add children of prepositions
                            if child.dep_ == "prep":
                                for grandchild in child.children:
                                    if grandchild.dep_ == "pobj":
                                        phrase_tokens.append(grandchild.text)

                    if len(phrase_tokens) >= 2:
                        verb_phrases.append(' '.join(phrase_tokens).lower())

        return list(set(verb_phrases))

    def extract_candidate_mwes(
        self,
        texts: List[str],
        min_length: int = 2,
        max_length: int = 6,
        min_freq: int = 2
    ) -> Dict[str, Dict]:
        """
        Extract candidate MWEs using multiple methods.

        Args:
            texts: List of text strings
            min_length: Minimum number of words
            max_length: Maximum number of words
            min_freq: Minimum frequency

        Returns:
            Dictionary of MWEs with metadata
        """
        candidates = {}

        # Extract n-grams for different values of n
        for n in range(min_length, max_length + 1):
            ngrams = self.extract_ngrams(texts, n=n, min_freq=min_freq)
            for ngram, freq in ngrams:
                if ngram not in candidates:
                    candidates[ngram] = {
                        "frequency": freq,
                        "length": n,
                        "type": "ngram"
                    }

        # Extract noun phrases
        noun_phrases = self.extract_noun_phrases(texts)
        for np in noun_phrases:
            if np not in candidates:
                candidates[np] = {
                    "frequency": texts.count(np),
                    "length": len(np.split()),
                    "type": "noun_phrase"
                }

        # Extract verb phrases
        verb_phrases = self.extract_verb_phrases(texts)
        for vp in verb_phrases:
            if vp not in candidates:
                candidates[vp] = {
                    "frequency": texts.count(vp),
                    "length": len(vp.split()),
                    "type": "verb_phrase"
                }

        return candidates
