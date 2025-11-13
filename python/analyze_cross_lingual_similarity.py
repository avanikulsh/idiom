"""
Analyze cross-lingual semantic similarity between English and French idioms.
Based on embeddings from usage contexts (symmetric representation).
"""
import pickle
import json
import csv
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_embeddings(embedding_file):
    """Load saved embeddings."""
    with open(embedding_file, 'rb') as f:
        data = pickle.load(f)
    return data

def main():
    print("=" * 80)
    print("CROSS-LINGUAL SEMANTIC SIMILARITY ANALYSIS")
    print("=" * 80)
    print("\nAnalyzing semantic similarity between English and French idioms")
    print("Based on embeddings from usage contexts\n")

    # Load embeddings
    emb_dir = Path("data/embeddings")

    print("Loading English embeddings...")
    en_data = load_embeddings(emb_dir / "english_idiom_embeddings.pkl")
    en_idioms = en_data['idioms']
    en_embeddings = en_data['embeddings']
    print(f"✓ Loaded {len(en_idioms):,} English idioms")

    print("\nLoading French embeddings...")
    fr_data = load_embeddings(emb_dir / "french_idiom_embeddings.pkl")
    fr_idioms = fr_data['idioms']
    fr_embeddings = fr_data['embeddings']
    print(f"✓ Loaded {len(fr_idioms):,} French idioms")

    # Compute cross-lingual similarity matrix
    print("\n" + "=" * 80)
    print("COMPUTING CROSS-LINGUAL SIMILARITY MATRIX")
    print("=" * 80)

    print(f"\nCalculating cosine similarities...")
    print(f"English idioms: {len(en_idioms):,}")
    print(f"French idioms:  {len(fr_idioms):,}")
    print(f"Matrix size: {len(en_idioms):,} × {len(fr_idioms):,} = {len(en_idioms) * len(fr_idioms):,} comparisons")

    similarities = cosine_similarity(en_embeddings, fr_embeddings)
    print(f"✓ Computed similarity matrix: {similarities.shape}")

    # Find top cross-lingual matches
    print("\n" + "=" * 80)
    print("TOP 50 CROSS-LINGUAL SEMANTIC SIMILARITIES")
    print("=" * 80)

    # Flatten and sort all similarities
    all_matches = []

    for en_idx in range(len(en_idioms)):
        for fr_idx in range(len(fr_idioms)):
            sim = similarities[en_idx, fr_idx]

            all_matches.append({
                'english_idiom': en_idioms[en_idx]['idiom'],
                'english_context': en_idioms[en_idx]['contexts'][0] if en_idioms[en_idx]['contexts'] else '',
                'french_idiom': fr_idioms[fr_idx]['idiom'],
                'french_context': fr_idioms[fr_idx]['contexts'][0] if fr_idioms[fr_idx]['contexts'] else '',
                'french_english_translation': fr_idioms[fr_idx]['english_translations'][0] if fr_idioms[fr_idx]['english_translations'] else '',
                'similarity': float(sim)
            })

    # Sort by similarity
    all_matches_sorted = sorted(all_matches, key=lambda x: x['similarity'], reverse=True)

    # Display top 50
    print("\nMost semantically similar idiom pairs across languages:\n")

    for i, match in enumerate(all_matches_sorted[:50], 1):
        print(f"{i:2d}. Similarity: {match['similarity']:.4f}")
        print(f"    EN: {match['english_idiom']}")
        print(f"    FR: {match['french_idiom']}")
        print(f"    EN context: {match['english_context'][:100]}...")
        print(f"    FR context: {match['french_context'][:100]}...")
        print(f"    FR→EN translation: {match['french_english_translation'][:100]}...")
        print()

    # Save top 100 matches
    print("=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    output_dir = Path("data/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    output_json = output_dir / "cross_lingual_semantic_similarities.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_matches_sorted[:100], f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved top 100 matches to: {output_json}")

    # Save CSV
    output_csv = output_dir / "cross_lingual_semantic_similarities.csv"
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_matches_sorted[0].keys())
        writer.writeheader()
        writer.writerows(all_matches_sorted[:100])
    print(f"✓ Saved to CSV: {output_csv}")

    # Analyze best match for each French idiom
    print("\n" + "=" * 80)
    print("BEST ENGLISH MATCH FOR EACH FRENCH IDIOM")
    print("=" * 80)

    fr_best_matches = []

    for fr_idx, fr_idiom in enumerate(fr_idioms):
        # Get similarities for this French idiom
        sims = similarities[:, fr_idx]

        # Find best English match
        best_en_idx = np.argmax(sims)
        best_sim = sims[best_en_idx]

        fr_best_matches.append({
            'french_idiom': fr_idiom['idiom'],
            'french_context': fr_idiom['contexts'][0] if fr_idiom['contexts'] else '',
            'french_english_translation': fr_idiom['english_translations'][0] if fr_idiom['english_translations'] else '',
            'best_english_match': en_idioms[best_en_idx]['idiom'],
            'english_context': en_idioms[best_en_idx]['contexts'][0] if en_idioms[best_en_idx]['contexts'] else '',
            'similarity': float(best_sim)
        })

    # Sort by similarity
    fr_best_matches_sorted = sorted(fr_best_matches, key=lambda x: x['similarity'], reverse=True)

    print("\nTop 20 French idioms with their best English semantic match:\n")

    for i, match in enumerate(fr_best_matches_sorted[:20], 1):
        print(f"{i:2d}. Similarity: {match['similarity']:.4f}")
        print(f"    FR: {match['french_idiom']}")
        print(f"    EN: {match['best_english_match']}")
        print(f"    FR→EN translation: {match['french_english_translation'][:80]}...")
        print()

    # Save
    output_fr_json = output_dir / "french_best_english_matches.json"
    with open(output_fr_json, 'w', encoding='utf-8') as f:
        json.dump(fr_best_matches_sorted, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved all French→English best matches to: {output_fr_json}")

    # Statistics
    print("\n" + "=" * 80)
    print("SIMILARITY STATISTICS")
    print("=" * 80)

    all_sims = similarities.flatten()
    fr_best_sims = [m['similarity'] for m in fr_best_matches]

    print("\nAll cross-lingual similarities:")
    print(f"  Mean:   {np.mean(all_sims):.4f}")
    print(f"  Median: {np.median(all_sims):.4f}")
    print(f"  Std:    {np.std(all_sims):.4f}")
    print(f"  Min:    {np.min(all_sims):.4f}")
    print(f"  Max:    {np.max(all_sims):.4f}")

    print("\nBest match for each French idiom:")
    print(f"  Mean:   {np.mean(fr_best_sims):.4f}")
    print(f"  Median: {np.median(fr_best_sims):.4f}")
    print(f"  Min:    {np.min(fr_best_sims):.4f}")
    print(f"  Max:    {np.max(fr_best_sims):.4f}")

    # Similarity distribution
    print("\nSimilarity distribution (all pairs):")
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    for thresh in thresholds:
        count = np.sum(all_sims >= thresh)
        pct = count / len(all_sims) * 100
        print(f"  >= {thresh:.1f}: {count:7,} ({pct:5.2f}%)")

    print("\nSimilarity distribution (best match per French idiom):")
    for thresh in thresholds:
        count = sum(1 for s in fr_best_sims if s >= thresh)
        pct = count / len(fr_best_sims) * 100
        print(f"  >= {thresh:.1f}: {count:4d} ({pct:5.1f}%)")

    return len(all_matches_sorted)


if __name__ == "__main__":
    try:
        count = main()
        print(f"\n✓ SUCCESS! Analyzed {count:,} cross-lingual idiom pairs")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
