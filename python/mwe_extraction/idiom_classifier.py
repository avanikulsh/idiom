"""
Classify Spanish MWEs as idiomatic or non-idiomatic.
Uses multiple strategies: heuristics, patterns, and semantic analysis.
"""
import re
from typing import List, Dict, Tuple, Set
from collections import Counter


class IdiomClassifier:
    """Classify MWEs as idiomatic or literal."""

    def __init__(self):
        """Initialize classifier with known patterns and stopwords."""

        # Spanish function words that appear in non-idiomatic phrases
        self.function_words = {
            'el', 'la', 'los', 'las',  # articles
            'un', 'una', 'unos', 'unas',  # indefinite articles
            'de', 'a', 'en', 'por', 'para', 'con', 'sin',  # prepositions
            'que', 'cual', 'quien', 'donde', 'cuando',  # relatives
            'no', 'sí', 'y', 'o', 'pero',  # conjunctions
            'es', 'está', 'son', 'están', 'ser', 'estar',  # copulas
            'muy', 'más', 'menos', 'tan', 'tanto',  # degree
            'me', 'te', 'se', 'le', 'lo', 'la',  # pronouns
            'mi', 'tu', 'su',  # possessives
        }

        # Patterns that indicate NON-idiomatic expressions
        self.non_idiomatic_patterns = [
            r'^no (lo|la|me|te|se|es|soy|eres)',  # no + pronoun/copula
            r'^(es|está|son|están) (un|una|el|la|muy)',  # copula + article/degree
            r'^(sí|no), (sí|no)',  # yes/no repetition
            r'^(por|para|de|a|en|con) (el|la|los|las)',  # prep + article
            r'^\?',  # question words alone
            r'^(lo|la|los|las) que$',  # simple relative clauses
            r'^voy a$',  # going to
            r'^tengo que$',  # have to
            r'^hay que$',  # must
        ]

        # Known Spanish idioms (seed list - would expand this)
        self.known_idioms = {
            'meter la pata', 'meter el pie',  # put your foot in it
            'tomar el pelo', 'tomarle el pelo',  # pull someone's leg
            'estar en las nubes',  # head in the clouds
            'llover a cántaros', 'llueve a cántaros',  # raining cats and dogs
            'costar un ojo de la cara',  # cost an arm and a leg
            'dar la vuelta a la tortilla',  # turn the tables
            'no tener pelos en la lengua',  # not mince words
            'entre la espada y la pared',  # between a rock and a hard place
            'dormirse en los laureles',  # rest on your laurels
            'buscar una aguja en un pajar',  # needle in a haystack
            'pan comido',  # piece of cake
            'tirar la casa por la ventana',  # spare no expense
            'ser pan comido',  # be a piece of cake
            'hacer caso',  # pay attention
            'dar la lata',  # be annoying
            'echar de menos',  # to miss
            'tener en cuenta',  # take into account
            'estar hasta las narices',  # be fed up
            'dar en el clavo',  # hit the nail on the head
            'ponerse las pilas',  # get one\'s act together
            'estar al loro',  # be on the ball
            'ser un rollo',  # be boring
            'hacer el ridículo',  # make a fool of oneself
            'estar en la luna',  # daydream
            'quedarse de piedra',  # be stunned
            'partir el corazón',  # break someone\'s heart
            'romper el hielo',  # break the ice
            'ver las estrellas',  # see stars (from pain)
            'perder la cabeza',  # lose one\'s head
            'dar la cara',  # face the music
            'poner los cuernos',  # cheat on someone
            'estar de moda',  # be in fashion
            'hacer caso omiso',  # ignore
        }

    def is_mostly_function_words(self, mwe: str) -> bool:
        """Check if MWE is mostly function words."""
        words = mwe.lower().split()
        if len(words) == 0:
            return True

        function_word_count = sum(1 for w in words if w in self.function_words)
        return function_word_count / len(words) > 0.7

    def matches_non_idiomatic_pattern(self, mwe: str) -> bool:
        """Check if MWE matches non-idiomatic patterns."""
        mwe_lower = mwe.lower()
        for pattern in self.non_idiomatic_patterns:
            if re.match(pattern, mwe_lower):
                return True
        return False

    def is_known_idiom(self, mwe: str) -> bool:
        """Check if MWE is a known Spanish idiom."""
        mwe_normalized = mwe.lower().strip('¿?¡!.,;:')
        return mwe_normalized in self.known_idioms

    def calculate_idiomaticity_score(self, mwe: str, frequency: int, length: int) -> float:
        """
        Calculate idiomaticity score for an MWE.

        Returns:
            Score from 0 (definitely not idiomatic) to 1 (likely idiomatic)
        """
        score = 0.5  # Start neutral

        # Known idiom - very high score
        if self.is_known_idiom(mwe):
            return 0.95

        # Length heuristics
        if length < 2:
            score -= 0.3  # Too short
        elif length >= 3:
            score += 0.1  # Longer phrases more likely idiomatic

        # Function word ratio
        if self.is_mostly_function_words(mwe):
            score -= 0.3

        # Non-idiomatic patterns
        if self.matches_non_idiomatic_pattern(mwe):
            score -= 0.4

        # Frequency heuristics (very high or very low can indicate idiom)
        if frequency < 5:
            score -= 0.1  # Rare might be noise
        elif 10 <= frequency <= 50:
            score += 0.1  # Moderate frequency good for idioms

        # Contains specific idiomatic markers
        idiomatic_markers = ['pata', 'pelo', 'nubes', 'piedra', 'corazón',
                            'cabeza', 'cara', 'ojos', 'manos', 'pies',
                            'narices', 'boca', 'lengua']

        words = mwe.lower().split()
        if any(marker in words for marker in idiomatic_markers):
            score += 0.15

        # Has verb + noun (common idiom structure)
        # This is a simplification - would need POS tagging for accuracy
        verb_markers = ['hacer', 'dar', 'poner', 'tener', 'estar', 'ser',
                       'meter', 'tomar', 'costar', 'llover', 'buscar', 'tirar']
        if any(verb in words for verb in verb_markers) and len(words) >= 3:
            score += 0.1

        return max(0.0, min(1.0, score))

    def classify_mwes(
        self,
        mwes: Dict[str, Dict],
        threshold: float = 0.6
    ) -> Dict[str, Dict]:
        """
        Classify MWEs as idiomatic or not.

        Args:
            mwes: Dictionary of {mwe: {frequency, length, type}}
            threshold: Idiomaticity score threshold

        Returns:
            Dictionary with added 'idiomaticity_score' and 'is_idiomatic' fields
        """
        classified = {}

        for mwe, info in mwes.items():
            score = self.calculate_idiomaticity_score(
                mwe,
                info.get('frequency', 0),
                info.get('length', 0)
            )

            classified[mwe] = {
                **info,
                'idiomaticity_score': score,
                'is_idiomatic': score >= threshold,
                'known_idiom': self.is_known_idiom(mwe)
            }

        return classified

    def get_idiomatic_candidates(
        self,
        mwes: Dict[str, Dict],
        threshold: float = 0.6,
        min_score: float = 0.5
    ) -> List[Tuple[str, Dict]]:
        """
        Get MWEs that are likely idiomatic.

        Args:
            mwes: Dictionary of MWEs
            threshold: Classification threshold
            min_score: Minimum score to include

        Returns:
            Sorted list of (mwe, info) tuples
        """
        classified = self.classify_mwes(mwes, threshold)

        candidates = [
            (mwe, info) for mwe, info in classified.items()
            if info['idiomaticity_score'] >= min_score
        ]

        # Sort by idiomaticity score (descending), then frequency
        candidates.sort(
            key=lambda x: (x[1]['idiomaticity_score'], x[1]['frequency']),
            reverse=True
        )

        return candidates
