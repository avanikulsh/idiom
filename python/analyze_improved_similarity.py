"""
Improved cross-lingual semantic similarity analysis.
Uses dual embeddings with weighted scoring and lexical overlap filtering.
"""
import pickle
import json
import csv
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter


def load_dual_embeddings(embedding_file):
    """Load saved dual embeddings."""
    with open(embedding_file, 'rb') as f:
        data = pickle.load(f)
    return data


def get_tokens(text):
    """Extract tokens from text (lowercase, alphanumeric only)."""
    # Remove punctuation and convert to lowercase
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return set(tokens)


def calculate_lexical_overlap(idiom1, idiom2):
    """
    Calculate lexical overlap ratio between two idioms.
    Returns ratio of shared tokens to total unique tokens.
    """
    tokens1 = get_tokens(idiom1)
    tokens2 = get_tokens(idiom2)

    if not tokens1 or not tokens2:
        return 0.0

    shared = tokens1.intersection(tokens2)
    total = tokens1.union(tokens2)

    return len(shared) / len(total) if total else 0.0


def compute_weighted_similarity(idiom_only_sim, context_sim,
                                idiom1, idiom2,
                                idiom_weight=0.6, context_weight=0.4,
                                lexical_penalty=True):
    """
    Compute weighted similarity with optional lexical overlap penalty.

    Args:
        idiom_only_sim: Similarity between idiom-only embeddings
        context_sim: Similarity between idiom+context embeddings
        idiom1, idiom2: The actual idiom texts
        idiom_weight: Weight for idiom-only similarity (default 0.6)
        context_weight: Weight for context similarity (default 0.4)
        lexical_penalty: Whether to penalize high lexical overlap (default True)

    Returns:
        Weighted similarity score (0-1)
    """
    # Base weighted score
    weighted_score = (idiom_weight * idiom_only_sim) + (context_weight * context_sim)

    # Apply lexical overlap penalty if enabled
    if lexical_penalty:
        overlap = calculate_lexical_overlap(idiom1, idiom2)

        # If overlap > 0.3 but similarity is high, penalize
        # This catches cases like "ear" matching all idioms with "ear"
        if overlap > 0.3 and weighted_score > 0.6:
            # Reduce score proportionally to overlap
            penalty_factor = 1 - (overlap * 0.5)  # Max 50% penalty
            weighted_score *= penalty_factor

    return weighted_score


def analyze_language_pair(en_idioms, en_embeddings,
                          target_idioms, target_embeddings,
                          lang_name, lang_code,
                          min_threshold=0.65,
                          idiom_weight=0.6,
                          context_weight=0.4):
    """
    Analyze cross-lingual similarity with improved scoring and filtering.
    """
    print("=" * 80)
    print(f"IMPROVED ANALYSIS: ENGLISH ↔ {lang_name.upper()}")
    print("=" * 80)

    print(f"\nConfiguration:")
    print(f"  Idiom-only weight: {idiom_weight:.1%}")
    print(f"  Context weight:    {context_weight:.1%}")
    print(f"  Min threshold:     {min_threshold:.2f}")
    print(f"  Lexical penalty:   Enabled")

    print(f"\nCalculating similarities...")
    print(f"English idioms: {len(en_idioms):,}")
    print(f"{lang_name} idioms: {len(target_idioms):,}")

    # Compute both similarity matrices
    idiom_only_sims = cosine_similarity(
        en_embeddings['idiom_only_embeddings'],
        target_embeddings['idiom_only_embeddings']
    )

    context_sims = cosine_similarity(
        en_embeddings['idiom_context_embeddings'],
        target_embeddings['idiom_context_embeddings']
    )

    print(f"✓ Computed dual similarity matrices: {idiom_only_sims.shape}")

    # Find top matches with weighted scoring
    print(f"\nComputing weighted scores with lexical filtering...")

    all_matches = []

    for en_idx in range(len(en_idioms)):
        en_idiom = en_idioms[en_idx]['idiom']

        for tgt_idx in range(len(target_idioms)):
            tgt_idiom = target_idioms[tgt_idx]['idiom']

            # Get base similarities
            idiom_sim = idiom_only_sims[en_idx, tgt_idx]
            context_sim = context_sims[en_idx, tgt_idx]

            # Compute weighted similarity with lexical penalty
            weighted_sim = compute_weighted_similarity(
                idiom_sim, context_sim,
                en_idiom, tgt_idiom,
                idiom_weight, context_weight,
                lexical_penalty=True
            )

            # Only include if above threshold
            if weighted_sim >= min_threshold:
                lexical_overlap = calculate_lexical_overlap(en_idiom, tgt_idiom)

                all_matches.append({
                    'english_idiom': en_idiom,
                    'english_context': en_idioms[en_idx]['contexts'][0] if en_idioms[en_idx]['contexts'] else '',
                    f'{lang_code}_idiom': tgt_idiom,
                    f'{lang_code}_context': target_idioms[tgt_idx]['contexts'][0] if target_idioms[tgt_idx]['contexts'] else '',
                    'english_translation': target_idioms[tgt_idx]['english_translations'][0] if target_idioms[tgt_idx]['english_translations'] else '',
                    'weighted_similarity': float(weighted_sim),
                    'idiom_only_similarity': float(idiom_sim),
                    'context_similarity': float(context_sim),
                    'lexical_overlap': float(lexical_overlap)
                })

    # Sort by weighted similarity
    all_matches_sorted = sorted(all_matches, key=lambda x: x['weighted_similarity'], reverse=True)

    print(f"✓ Found {len(all_matches_sorted):,} matches above threshold {min_threshold:.2f}")

    # Display top 30
    print(f"\n{'=' * 80}")
    print(f"TOP 30 MATCHES (WEIGHTED SCORING)")
    print("=" * 80)
    print()

    for i, match in enumerate(all_matches_sorted[:30], 1):
        print(f"{i:2d}. Weighted: {match['weighted_similarity']:.4f} | Idiom: {match['idiom_only_similarity']:.4f} | Context: {match['context_similarity']:.4f} | Overlap: {match['lexical_overlap']:.2%}")
        print(f"    EN: {match['english_idiom']}")
        print(f"    {lang_code.upper()}: {match[f'{lang_code}_idiom']}")
        print(f"    {lang_code.upper()}→EN: {match['english_translation'][:100]}...")
        print()

    # Save results
    output_dir = Path("data/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_json = output_dir / f"improved_{lang_code}_matches.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_matches_sorted[:100], f, indent=2, ensure_ascii=False)
    print(f"✓ Saved top 100 improved matches to: {output_json}")

    output_csv = output_dir / f"improved_{lang_code}_matches.csv"
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        if all_matches_sorted:
            writer = csv.DictWriter(f, fieldnames=all_matches_sorted[0].keys())
            writer.writeheader()
            writer.writerows(all_matches_sorted[:100])
    print(f"✓ Saved to CSV: {output_csv}")

    # Find best match for each target idiom
    print(f"\n{'=' * 80}")
    print(f"BEST ENGLISH MATCH PER {lang_name.upper()} IDIOM")
    print("=" * 80)

    target_best_matches = []

    for tgt_idx, tgt_idiom_data in enumerate(target_idioms):
        tgt_idiom = tgt_idiom_data['idiom']

        best_score = -1
        best_en_idx = -1
        best_idiom_sim = 0
        best_context_sim = 0

        for en_idx in range(len(en_idioms)):
            en_idiom = en_idioms[en_idx]['idiom']

            idiom_sim = idiom_only_sims[en_idx, tgt_idx]
            context_sim = context_sims[en_idx, tgt_idx]

            weighted_sim = compute_weighted_similarity(
                idiom_sim, context_sim,
                en_idiom, tgt_idiom,
                idiom_weight, context_weight,
                lexical_penalty=True
            )

            if weighted_sim > best_score:
                best_score = weighted_sim
                best_en_idx = en_idx
                best_idiom_sim = idiom_sim
                best_context_sim = context_sim

        lexical_overlap = calculate_lexical_overlap(
            en_idioms[best_en_idx]['idiom'],
            tgt_idiom
        )

        target_best_matches.append({
            f'{lang_code}_idiom': tgt_idiom,
            f'{lang_code}_context': tgt_idiom_data['contexts'][0] if tgt_idiom_data['contexts'] else '',
            'english_translation': tgt_idiom_data['english_translations'][0] if tgt_idiom_data['english_translations'] else '',
            'best_english_match': en_idioms[best_en_idx]['idiom'],
            'english_context': en_idioms[best_en_idx]['contexts'][0] if en_idioms[best_en_idx]['contexts'] else '',
            'weighted_similarity': float(best_score),
            'idiom_only_similarity': float(best_idiom_sim),
            'context_similarity': float(best_context_sim),
            'lexical_overlap': float(lexical_overlap)
        })

    # Sort by weighted similarity
    target_best_matches_sorted = sorted(target_best_matches, key=lambda x: x['weighted_similarity'], reverse=True)

    print(f"\nTop 20 {lang_name} idioms with best English match:\n")

    for i, match in enumerate(target_best_matches_sorted[:20], 1):
        print(f"{i:2d}. Weighted: {match['weighted_similarity']:.4f} | Idiom: {match['idiom_only_similarity']:.4f} | Context: {match['context_similarity']:.4f}")
        print(f"    {lang_code.upper()}: {match[f'{lang_code}_idiom']}")
        print(f"    EN: {match['best_english_match']}")
        print(f"    {lang_code.upper()}→EN: {match['english_translation'][:80]}...")
        print()

    # Save
    output_best_json = output_dir / f"improved_{lang_code}_best_matches.json"
    with open(output_best_json, 'w', encoding='utf-8') as f:
        json.dump(target_best_matches_sorted, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved all {lang_name}→English best matches to: {output_best_json}")

    # Statistics
    print(f"\n{'=' * 80}")
    print("QUALITY METRICS")
    print("=" * 80)

    # Analyze matches above threshold
    weighted_scores = [m['weighted_similarity'] for m in all_matches_sorted]
    idiom_scores = [m['idiom_only_similarity'] for m in all_matches_sorted]
    context_scores = [m['context_similarity'] for m in all_matches_sorted]
    overlaps = [m['lexical_overlap'] for m in all_matches_sorted]

    if weighted_scores:
        print(f"\nMatches above threshold ({min_threshold:.2f}):")
        print(f"  Count: {len(weighted_scores):,}")
        print(f"  Weighted similarity:  Mean={np.mean(weighted_scores):.4f}, Median={np.median(weighted_scores):.4f}")
        print(f"  Idiom-only similarity: Mean={np.mean(idiom_scores):.4f}, Median={np.median(idiom_scores):.4f}")
        print(f"  Context similarity:    Mean={np.mean(context_scores):.4f}, Median={np.median(context_scores):.4f}")
        print(f"  Lexical overlap:       Mean={np.mean(overlaps):.2%}, Median={np.median(overlaps):.2%}")

        # High lexical overlap warnings
        high_overlap = [m for m in all_matches_sorted if m['lexical_overlap'] > 0.4]
        if high_overlap:
            print(f"\n  ⚠️  {len(high_overlap)} matches have >40% lexical overlap (may be spurious)")

    # Best match statistics
    best_weighted = [m['weighted_similarity'] for m in target_best_matches_sorted]

    print(f"\nBest match per {lang_name} idiom:")
    print(f"  Mean weighted similarity: {np.mean(best_weighted):.4f}")
    print(f"  Median:                   {np.median(best_weighted):.4f}")
    print(f"  Min:                      {np.min(best_weighted):.4f}")
    print(f"  Max:                      {np.max(best_weighted):.4f}")

    # Distribution
    print(f"\nWeighted similarity distribution (best match per {lang_name} idiom):")
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    for thresh in thresholds:
        count = sum(1 for s in best_weighted if s >= thresh)
        pct = count / len(best_weighted) * 100
        print(f"  >= {thresh:.1f}: {count:4d} ({pct:5.1f}%)")

    return len(all_matches_sorted), target_best_matches_sorted


def main():
    print("=" * 80)
    print("IMPROVED CROSS-LINGUAL IDIOM MATCHING")
    print("=" * 80)
    print("\nImprovements:")
    print("  ✓ Dual embeddings (idiom-only + idiom+context)")
    print("  ✓ Weighted scoring (60% idiom, 40% context)")
    print("  ✓ Lexical overlap penalty (reduces spurious matches)")
    print("  ✓ Higher quality threshold (0.65+)")
    print()

    # Load dual embeddings
    emb_dir = Path("data/embeddings")

    print("Loading dual embeddings...")
    en_data = load_dual_embeddings(emb_dir / "english_dual_embeddings.pkl")
    en_idioms = en_data['idioms']
    print(f"✓ English: {len(en_idioms):,} idioms")

    fr_data = load_dual_embeddings(emb_dir / "french_dual_embeddings.pkl")
    fr_idioms = fr_data['idioms']
    print(f"✓ French: {len(fr_idioms):,} idioms")

    fi_data = load_dual_embeddings(emb_dir / "finnish_dual_embeddings.pkl")
    fi_idioms = fi_data['idioms']
    print(f"✓ Finnish: {len(fi_idioms):,} idioms")

    jp_data = load_dual_embeddings(emb_dir / "japanese_dual_embeddings.pkl")
    jp_idioms = jp_data['idioms']
    print(f"✓ Japanese: {len(jp_idioms):,} idioms")
    print()

    # Analyze all language pairs
    results = {}

    # French
    fr_count, fr_matches = analyze_language_pair(
        en_idioms, en_data,
        fr_idioms, fr_data,
        "French", "fr",
        min_threshold=0.65
    )
    results['French'] = (fr_count, fr_matches)

    print("\n\n")

    # Finnish
    fi_count, fi_matches = analyze_language_pair(
        en_idioms, en_data,
        fi_idioms, fi_data,
        "Finnish", "fi",
        min_threshold=0.65
    )
    results['Finnish'] = (fi_count, fi_matches)

    print("\n\n")

    # Japanese
    jp_count, jp_matches = analyze_language_pair(
        en_idioms, en_data,
        jp_idioms, jp_data,
        "Japanese", "jp",
        min_threshold=0.65
    )
    results['Japanese'] = (jp_count, jp_matches)

    # Summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)

    print("\nMatches above threshold (0.65) by language:")
    for lang, (count, _) in results.items():
        print(f"  {lang:10s}: {count:5,} high-quality matches")

    total_matches = sum(count for count, _ in results.values())
    print(f"\nTotal high-quality matches: {total_matches:,}")

    print("\nExpected improvement:")
    print("  ✓ Fewer spurious lexical matches (ear→ear, head→head)")
    print("  ✓ Better metaphorical structure matching")
    print("  ✓ Reduced context-only matches")
    print("  ✓ Higher precision (fewer false positives)")

    return results


if __name__ == "__main__":
    try:
        results = main()
        total = sum(count for count, _ in results.values())
        print(f"\n✓ SUCCESS! Generated {total:,} improved cross-lingual matches")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
