"""
Semantic similarity matching for cross-lingual idiom comparison.
"""
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Tuple
import torch


class SemanticMatcher:
    """Match idioms across languages using multilingual embeddings."""

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        """
        Initialize the semantic matcher.

        Args:
            model_name: Name of the sentence transformer model to use
        """
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Model loaded on {self.device}")

    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode texts into embeddings.

        Args:
            texts: List of text strings
            batch_size: Batch size for encoding

        Returns:
            Numpy array of embeddings
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings

    def compute_similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between two sets of embeddings.

        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings

        Returns:
            Similarity matrix (shape: len(embeddings1) x len(embeddings2))
        """
        return cosine_similarity(embeddings1, embeddings2)

    def encode_idioms_with_context(
        self,
        idiom_data: List[Dict],
        use_contexts: bool = True,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Encode idioms using their contextual examples for better semantic representation.

        Args:
            idiom_data: List of idiom dictionaries with 'text' and optional 'contexts'
            use_contexts: Whether to use contextual examples
            batch_size: Batch size for encoding

        Returns:
            Numpy array of embeddings
        """
        texts_to_encode = []

        for item in idiom_data:
            idiom_text = item.get('text', item.get('idiom', ''))

            if use_contexts and 'contexts' in item and item['contexts']:
                # Combine idiom with its context for richer semantic representation
                # Format: "idiom: context_sentence"
                contexts = item['contexts'][:3]  # Use up to 3 contexts
                combined = f"{idiom_text}: {' '.join(contexts)}"
                texts_to_encode.append(combined)
            else:
                texts_to_encode.append(idiom_text)

        return self.encode_texts(texts_to_encode, batch_size=batch_size)

    def find_similar_mwes(
        self,
        english_idioms: List[str],
        foreign_mwes: List[str],
        threshold: float = 0.6,
        top_k: int = 5
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Find similar MWEs in foreign language for each English idiom.

        Args:
            english_idioms: List of English idioms (or idiom dicts with contexts)
            foreign_mwes: List of foreign language MWEs
            threshold: Minimum similarity threshold
            top_k: Number of top matches to return per idiom

        Returns:
            Dictionary mapping each English idiom to list of (foreign_mwe, similarity_score)
        """
        # Handle both string lists and dict lists
        if isinstance(english_idioms[0], dict):
            print(f"Encoding {len(english_idioms)} English idioms with contexts...")
            idiom_embeddings = self.encode_idioms_with_context(english_idioms, use_contexts=True)
            idiom_texts = [item.get('text', item.get('idiom', '')) for item in english_idioms]
        else:
            print(f"Encoding {len(english_idioms)} English idioms...")
            idiom_embeddings = self.encode_texts(english_idioms)
            idiom_texts = english_idioms

        print(f"Encoding {len(foreign_mwes)} foreign MWEs...")
        mwe_embeddings = self.encode_texts(foreign_mwes)

        print("Computing similarity matrix...")
        similarity_matrix = self.compute_similarity(idiom_embeddings, mwe_embeddings)

        # Find top matches for each idiom
        matches = {}

        for i, idiom in enumerate(idiom_texts):
            # Get similarities for this idiom
            similarities = similarity_matrix[i]

            # Get indices of top-k similar MWEs
            top_indices = np.argsort(similarities)[::-1][:top_k]

            # Filter by threshold and create match list
            matched_mwes = []
            for idx in top_indices:
                if similarities[idx] >= threshold:
                    matched_mwes.append((foreign_mwes[idx], float(similarities[idx])))

            if matched_mwes:
                matches[idiom] = matched_mwes

        return matches

    def find_best_match(
        self,
        query: str,
        candidates: List[str]
    ) -> Tuple[str, float]:
        """
        Find the best matching candidate for a query.

        Args:
            query: Query text
            candidates: List of candidate texts

        Returns:
            Tuple of (best_match, similarity_score)
        """
        query_embedding = self.encode_texts([query])
        candidate_embeddings = self.encode_texts(candidates)

        similarities = self.compute_similarity(query_embedding, candidate_embeddings)[0]
        best_idx = np.argmax(similarities)

        return candidates[best_idx], float(similarities[best_idx])

    def batch_match(
        self,
        queries: List[str],
        candidates: List[str],
        threshold: float = 0.6
    ) -> List[Dict]:
        """
        Batch matching of queries against candidates.

        Args:
            queries: List of query texts
            candidates: List of candidate texts
            threshold: Minimum similarity threshold

        Returns:
            List of match dictionaries with query, best_match, and score
        """
        query_embeddings = self.encode_texts(queries)
        candidate_embeddings = self.encode_texts(candidates)

        similarity_matrix = self.compute_similarity(query_embeddings, candidate_embeddings)

        results = []
        for i, query in enumerate(queries):
            best_idx = np.argmax(similarity_matrix[i])
            best_score = float(similarity_matrix[i][best_idx])

            if best_score >= threshold:
                results.append({
                    "query": query,
                    "best_match": candidates[best_idx],
                    "score": best_score
                })

        return results
