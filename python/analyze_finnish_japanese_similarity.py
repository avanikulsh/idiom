"""
Analyze cross-lingual semantic similarity for Finnish and Japanese with English.
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

def analyze_language_pair(en_idioms, en_embeddings, target_idioms, target_embeddings, lang_name, lang_code):
    """Analyze cross-lingual similarity between English and target language."""

    print("=" * 80)
    print(f"ENGLISH ↔ {lang_name.upper()} SEMANTIC SIMILARITY")
    print("=" * 80)

    print(f"\nCalculating cosine similarities...")
    print(f"English idioms: {len(en_idioms):,}")
    print(f"{lang_name} idioms: {len(target_idioms):,}")
    print(f"Matrix size: {len(en_idioms):,} × {len(target_idioms):,} = {len(en_idioms) * len(target_idioms):,} comparisons")

    similarities = cosine_similarity(en_embeddings, target_embeddings)
    print(f"✓ Computed similarity matrix: {similarities.shape}")

    # Find top matches
    print(f"\n{'=' * 80}")
    print(f"TOP 30 ENGLISH ↔ {lang_name.upper()} MATCHES")
    print("=" * 80)

    # Flatten and sort all similarities
    all_matches = []

    for en_idx in range(len(en_idioms)):
        for tgt_idx in range(len(target_idioms)):
            sim = similarities[en_idx, tgt_idx]

            all_matches.append({
                'english_idiom': en_idioms[en_idx]['idiom'],
                'english_context': en_idioms[en_idx]['contexts'][0] if en_idioms[en_idx]['contexts'] else '',
                f'{lang_code}_idiom': target_idioms[tgt_idx]['idiom'],
                f'{lang_code}_context': target_idioms[tgt_idx]['contexts'][0] if target_idioms[tgt_idx]['contexts'] else '',
                'english_translation': target_idioms[tgt_idx]['english_translations'][0] if target_idioms[tgt_idx]['english_translations'] else '',
                'similarity': float(sim)
            })

    # Sort by similarity
    all_matches_sorted = sorted(all_matches, key=lambda x: x['similarity'], reverse=True)

    # Display top 30
    print(f"\nMost semantically similar English ↔ {lang_name} idiom pairs:\n")

    for i, match in enumerate(all_matches_sorted[:30], 1):
        print(f"{i:2d}. Similarity: {match['similarity']:.4f}")
        print(f"    EN: {match['english_idiom']}")
        print(f"    {lang_code.upper()}: {match[f'{lang_code}_idiom']}")
        print(f"    EN context: {match['english_context'][:80]}...")
        print(f"    {lang_code.upper()} context: {match[f'{lang_code}_context'][:80]}...")
        print(f"    {lang_code.upper()}→EN translation: {match['english_translation'][:80]}...")
        print()

    # Save results
    output_dir = Path("data/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_json = output_dir / f"english_{lang_code}_similarities.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_matches_sorted[:100], f, indent=2, ensure_ascii=False)
    print(f"✓ Saved top 100 matches to: {output_json}")

    output_csv = output_dir / f"english_{lang_code}_similarities.csv"
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_matches_sorted[0].keys())
        writer.writeheader()
        writer.writerows(all_matches_sorted[:100])
    print(f"✓ Saved to CSV: {output_csv}")

    # Find best English match for each target language idiom
    print(f"\n{'=' * 80}")
    print(f"BEST ENGLISH MATCH FOR EACH {lang_name.upper()} IDIOM")
    print("=" * 80)

    target_best_matches = []

    for tgt_idx, tgt_idiom in enumerate(target_idioms):
        # Get similarities for this target idiom
        sims = similarities[:, tgt_idx]

        # Find best English match
        best_en_idx = np.argmax(sims)
        best_sim = sims[best_en_idx]

        target_best_matches.append({
            f'{lang_code}_idiom': tgt_idiom['idiom'],
            f'{lang_code}_context': tgt_idiom['contexts'][0] if tgt_idiom['contexts'] else '',
            'english_translation': tgt_idiom['english_translations'][0] if tgt_idiom['english_translations'] else '',
            'best_english_match': en_idioms[best_en_idx]['idiom'],
            'english_context': en_idioms[best_en_idx]['contexts'][0] if en_idioms[best_en_idx]['contexts'] else '',
            'similarity': float(best_sim)
        })

    # Sort by similarity
    target_best_matches_sorted = sorted(target_best_matches, key=lambda x: x['similarity'], reverse=True)

    print(f"\nTop 20 {lang_name} idioms with their best English semantic match:\n")

    for i, match in enumerate(target_best_matches_sorted[:20], 1):
        print(f"{i:2d}. Similarity: {match['similarity']:.4f}")
        print(f"    {lang_code.upper()}: {match[f'{lang_code}_idiom']}")
        print(f"    EN: {match['best_english_match']}")
        print(f"    {lang_code.upper()}→EN translation: {match['english_translation'][:80]}...")
        print()

    # Save
    output_best_json = output_dir / f"{lang_code}_best_english_matches.json"
    with open(output_best_json, 'w', encoding='utf-8') as f:
        json.dump(target_best_matches_sorted, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved all {lang_name}→English best matches to: {output_best_json}")

    # Statistics
    print(f"\n{'=' * 80}")
    print(f"SIMILARITY STATISTICS")
    print("=" * 80)

    all_sims = similarities.flatten()
    target_best_sims = [m['similarity'] for m in target_best_matches]

    print(f"\nAll English ↔ {lang_name} similarities:")
    print(f"  Mean:   {np.mean(all_sims):.4f}")
    print(f"  Median: {np.median(all_sims):.4f}")
    print(f"  Std:    {np.std(all_sims):.4f}")
    print(f"  Min:    {np.min(all_sims):.4f}")
    print(f"  Max:    {np.max(all_sims):.4f}")

    print(f"\nBest match for each {lang_name} idiom:")
    print(f"  Mean:   {np.mean(target_best_sims):.4f}")
    print(f"  Median: {np.median(target_best_sims):.4f}")
    print(f"  Min:    {np.min(target_best_sims):.4f}")
    print(f"  Max:    {np.max(target_best_sims):.4f}")

    # Similarity distribution
    print(f"\nSimilarity distribution (best match per {lang_name} idiom):")
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    for thresh in thresholds:
        count = sum(1 for s in target_best_sims if s >= thresh)
        pct = count / len(target_best_sims) * 100
        print(f"  >= {thresh:.1f}: {count:4d} ({pct:5.1f}%)")

    return len(all_matches_sorted), target_best_matches_sorted


def main():
    print("=" * 80)
    print("FINNISH & JAPANESE CROSS-LINGUAL SEMANTIC SIMILARITY")
    print("=" * 80)
    print("\nAnalyzing semantic similarity with English idioms")
    print("Based on embeddings from usage contexts\n")

    # Load embeddings
    emb_dir = Path("data/embeddings")

    print("Loading English embeddings...")
    en_data = load_embeddings(emb_dir / "english_idiom_embeddings.pkl")
    en_idioms = en_data['idioms']
    en_embeddings = en_data['embeddings']
    print(f"✓ Loaded {len(en_idioms):,} English idioms")

    print("\nLoading Finnish embeddings...")
    fi_data = load_embeddings(emb_dir / "finnish_idiom_embeddings.pkl")
    fi_idioms = fi_data['idioms']
    fi_embeddings = fi_data['embeddings']
    print(f"✓ Loaded {len(fi_idioms):,} Finnish idioms")

    print("\nLoading Japanese embeddings...")
    jp_data = load_embeddings(emb_dir / "japanese_idiom_embeddings.pkl")
    jp_idioms = jp_data['idioms']
    jp_embeddings = jp_data['embeddings']
    print(f"✓ Loaded {len(jp_idioms):,} Japanese idioms\n")

    # Analyze Finnish
    fi_count, fi_matches = analyze_language_pair(
        en_idioms, en_embeddings,
        fi_idioms, fi_embeddings,
        "Finnish", "fi"
    )

    print("\n\n")

    # Analyze Japanese
    jp_count, jp_matches = analyze_language_pair(
        en_idioms, en_embeddings,
        jp_idioms, jp_embeddings,
        "Japanese", "jp"
    )

    # Summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)

    print(f"\nEnglish ↔ Finnish: {fi_count:,} pairs analyzed")
    print(f"English ↔ Japanese: {jp_count:,} pairs analyzed")
    print(f"Total cross-lingual pairs: {fi_count + jp_count:,}")

    return fi_count, jp_count


if __name__ == "__main__":
    try:
        fi_count, jp_count = main()
        print(f"\n✓ SUCCESS! Analyzed {fi_count + jp_count:,} cross-lingual idiom pairs")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
