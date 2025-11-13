"""
Cross-lingual semantic matching between English and French idioms.
Both languages use SYMMETRIC representations: idiom + usage contexts.

English: MAGPIE idioms with BNC contexts
French: Crossing the Threshold idioms with movie subtitle contexts
"""
import json
import csv
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_english_idioms(magpie_file):
    """Load English idioms from MAGPIE with contexts."""
    print(f"\nLoading English idioms from: {magpie_file}")

    with open(magpie_file, 'r', encoding='utf-8') as f:
        magpie_data = json.load(f)

    english_idioms = []
    for item in magpie_data:
        contexts = [ex.get('sentence', '') for ex in item.get('examples', [])]

        if contexts:  # Only include idioms with contexts
            english_idioms.append({
                'idiom': item['idiom'],
                'contexts': contexts,
                'source': 'MAGPIE'
            })

    print(f"✓ Loaded {len(english_idioms):,} English idioms with contexts")
    return english_idioms


def load_french_idioms(french_file):
    """Load French idioms with contexts."""
    print(f"\nLoading French idioms from: {french_file}")

    french_idioms = []

    with open(french_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            french_contexts = row['french_contexts'].split(' ||| ')

            french_idioms.append({
                'idiom': row['idiom'],
                'contexts': french_contexts,
                'num_contexts': int(row['num_contexts']),
                'english_translations': row['english_translations'].split(' ||| ')
            })

    print(f"✓ Loaded {len(french_idioms):,} French idioms with contexts")
    return french_idioms


def create_idiom_representation(idiom, contexts, max_contexts=3):
    """Create text representation: idiom + contexts."""
    context_sample = contexts[:max_contexts]
    return f"{idiom}. " + " ".join(context_sample)


def main():
    print("=" * 80)
    print("CROSS-LINGUAL ENGLISH-FRENCH IDIOM MATCHING")
    print("=" * 80)
    print("\nSymmetric Representation: Both languages use idiom + usage contexts")
    print("English: MAGPIE idioms with BNC contexts")
    print("French: Crossing the Threshold idioms with movie subtitle contexts")

    # Load data
    magpie_file = Path("data/raw/english_idioms/magpie_idioms_with_context.json")
    french_file = Path("data/processed/french_idioms_with_contexts.csv")

    english_idioms = load_english_idioms(magpie_file)
    french_idioms = load_french_idioms(french_file)

    # Load multilingual sentence transformer
    print("\n" + "=" * 80)
    print("LOADING MULTILINGUAL SENTENCE TRANSFORMER")
    print("=" * 80)
    print("\nModel: paraphrase-multilingual-mpnet-base-v2")

    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    print("✓ Model loaded")

    # Create representations
    print("\n" + "=" * 80)
    print("CREATING EMBEDDINGS")
    print("=" * 80)

    print("\nCreating English representations...")
    english_texts = [create_idiom_representation(item['idiom'], item['contexts'])
                     for item in english_idioms]

    print(f"Sample English representation:\n  {english_texts[0][:150]}...\n")

    print("Encoding English idioms...")
    english_embeddings = model.encode(english_texts, show_progress_bar=True)
    print(f"✓ Encoded {len(english_embeddings):,} English idioms")

    print("\nCreating French representations...")
    french_texts = [create_idiom_representation(item['idiom'], item['contexts'])
                    for item in french_idioms]

    print(f"Sample French representation:\n  {french_texts[0][:150]}...\n")

    print("Encoding French idioms...")
    french_embeddings = model.encode(french_texts, show_progress_bar=True)
    print(f"✓ Encoded {len(french_embeddings):,} French idioms")

    # Compute similarities
    print("\n" + "=" * 80)
    print("COMPUTING CROSS-LINGUAL SIMILARITIES")
    print("=" * 80)

    print("\nCalculating cosine similarities...")
    similarities = cosine_similarity(english_embeddings, french_embeddings)
    print(f"✓ Similarity matrix shape: {similarities.shape}")

    # Find best matches for each English idiom
    print("\n" + "=" * 80)
    print("FINDING BEST MATCHES")
    print("=" * 80)

    matches = []

    for i, eng_idiom in enumerate(english_idioms):
        # Get top 5 matches for this English idiom
        sim_scores = similarities[i]
        top_indices = np.argsort(sim_scores)[-5:][::-1]

        for rank, fr_idx in enumerate(top_indices, 1):
            matches.append({
                'english_idiom': eng_idiom['idiom'],
                'english_context': eng_idiom['contexts'][0] if eng_idiom['contexts'] else '',
                'french_idiom': french_idioms[fr_idx]['idiom'],
                'french_context': french_idioms[fr_idx]['contexts'][0] if french_idioms[fr_idx]['contexts'] else '',
                'english_translation': french_idioms[fr_idx]['english_translations'][0] if french_idioms[fr_idx]['english_translations'] else '',
                'similarity': float(sim_scores[fr_idx]),
                'rank': rank
            })

    print(f"✓ Generated {len(matches):,} matches")

    # Show top matches
    print("\n" + "=" * 80)
    print("TOP 20 CROSS-LINGUAL MATCHES")
    print("=" * 80)

    # Sort by similarity
    matches_sorted = sorted(matches, key=lambda x: x['similarity'], reverse=True)

    for i, match in enumerate(matches_sorted[:20], 1):
        print(f"\n{i:2d}. Similarity: {match['similarity']:.3f}")
        print(f"    English: {match['english_idiom']}")
        print(f"    French:  {match['french_idiom']}")
        print(f"    EN Context: {match['english_context'][:80]}...")
        print(f"    FR Context: {match['french_context'][:80]}...")
        print(f"    FR→EN Translation: {match['english_translation'][:80]}...")

    # Save results
    output_json = Path("data/results/english_french_matches_symmetric.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    # Save top match for each English idiom
    top_matches = [m for m in matches if m['rank'] == 1]

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(top_matches, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved {len(top_matches):,} top matches to: {output_json}")

    # Save to CSV for easier viewing
    output_csv = Path("data/results/english_french_matches_symmetric.csv")

    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=top_matches[0].keys())
        writer.writeheader()
        writer.writerows(top_matches)

    print(f"✓ Saved to CSV: {output_csv}")

    # Statistics
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)

    similarities_top = [m['similarity'] for m in top_matches]

    print(f"\nTotal English idioms: {len(english_idioms):,}")
    print(f"Total French idioms: {len(french_idioms):,}")
    print(f"Matches found: {len(top_matches):,}")
    print(f"\nSimilarity distribution:")
    print(f"  Mean:   {np.mean(similarities_top):.3f}")
    print(f"  Median: {np.median(similarities_top):.3f}")
    print(f"  Min:    {np.min(similarities_top):.3f}")
    print(f"  Max:    {np.max(similarities_top):.3f}")

    # Count by similarity threshold
    thresholds = [0.5, 0.6, 0.7, 0.8]
    print(f"\nMatches by similarity threshold:")
    for thresh in thresholds:
        count = sum(1 for s in similarities_top if s >= thresh)
        pct = count / len(similarities_top) * 100
        print(f"  >= {thresh:.1f}: {count:4d} ({pct:5.1f}%)")

    return len(top_matches)


if __name__ == "__main__":
    try:
        count = main()
        print(f"\n✓ SUCCESS! Matched {count:,} English-French idiom pairs")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
