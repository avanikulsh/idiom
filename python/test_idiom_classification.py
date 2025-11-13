"""
Test idiom classification on extracted Spanish MWEs.
"""
import sys
import os
import csv
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from mwe_extraction.idiom_classifier import IdiomClassifier
from config import PROCESSED_DATA_DIR

def test_idiom_classification():
    """Test idiom classification on Spanish MWEs."""

    print("="*70)
    print("Spanish MWE Idiomaticity Classification")
    print("="*70)

    # Load extracted MWEs
    mwe_file = PROCESSED_DATA_DIR / "spanish_mwes.csv"

    print(f"\nLoading MWEs from: {mwe_file}")

    mwes = {}
    with open(mwe_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mwe = row['mwe']
            mwes[mwe] = {
                'frequency': int(row['frequency']),
                'length': int(row['length']),
                'type': row['type']
            }

    print(f"✓ Loaded {len(mwes):,} MWEs")

    # Initialize classifier
    print("\nInitializing idiom classifier...")
    classifier = IdiomClassifier()
    print(f"✓ Loaded {len(classifier.known_idioms)} known Spanish idioms")

    # Classify MWEs
    print("\nClassifying MWEs as idiomatic or literal...")
    classified = classifier.classify_mwes(mwes, threshold=0.6)

    # Get statistics
    idiomatic_count = sum(1 for info in classified.values() if info['is_idiomatic'])
    known_count = sum(1 for info in classified.values() if info['known_idiom'])

    print(f"\n✓ Classification complete")
    print(f"  Total MWEs: {len(classified):,}")
    print(f"  Classified as idiomatic: {idiomatic_count:,} ({idiomatic_count/len(classified)*100:.1f}%)")
    print(f"  Known idioms found: {known_count}")

    # Get idiomatic candidates
    candidates = classifier.get_idiomatic_candidates(mwes, threshold=0.6, min_score=0.5)

    print("\n" + "="*70)
    print("TOP 50 IDIOMATIC CANDIDATES")
    print("="*70)
    print(f"{'Rank':<5} {'MWE':<40} {'Score':<7} {'Freq':<6} {'Known?'}")
    print("-"*70)

    for i, (mwe, info) in enumerate(candidates[:50], 1):
        known_mark = "✓" if info['known_idiom'] else ""
        print(f"{i:<5} {mwe:<40} {info['idiomaticity_score']:.3f}   {info['frequency']:<6} {known_mark}")

    # Show breakdown by score range
    print("\n" + "="*70)
    print("SCORE DISTRIBUTION")
    print("="*70)

    score_ranges = {
        'Very likely idiomatic (0.8-1.0)': (0.8, 1.0),
        'Likely idiomatic (0.6-0.8)': (0.6, 0.8),
        'Possibly idiomatic (0.5-0.6)': (0.5, 0.6),
        'Probably literal (0.3-0.5)': (0.3, 0.5),
        'Definitely literal (0.0-0.3)': (0.0, 0.3),
    }

    for label, (min_score, max_score) in score_ranges.items():
        count = sum(1 for info in classified.values()
                   if min_score <= info['idiomaticity_score'] < max_score)
        print(f"  {label}: {count:,}")

    # Show some examples from each category
    print("\n" + "="*70)
    print("EXAMPLES BY CATEGORY")
    print("="*70)

    # Very likely idiomatic
    print("\n📌 VERY LIKELY IDIOMATIC (score >= 0.8):")
    very_likely = [(mwe, info) for mwe, info in classified.items()
                   if info['idiomaticity_score'] >= 0.8]
    very_likely.sort(key=lambda x: x[1]['idiomaticity_score'], reverse=True)

    for mwe, info in very_likely[:15]:
        mark = " [KNOWN]" if info['known_idiom'] else ""
        print(f"  • {mwe:<40} (score: {info['idiomaticity_score']:.3f}, freq: {info['frequency']:3d}){mark}")

    if not very_likely:
        print("  (none found)")

    # Likely idiomatic
    print("\n⭐ LIKELY IDIOMATIC (score 0.6-0.8):")
    likely = [(mwe, info) for mwe, info in classified.items()
              if 0.6 <= info['idiomaticity_score'] < 0.8]
    likely.sort(key=lambda x: (x[1]['idiomaticity_score'], x[1]['frequency']), reverse=True)

    for mwe, info in likely[:15]:
        print(f"  • {mwe:<40} (score: {info['idiomaticity_score']:.3f}, freq: {info['frequency']:3d})")

    if not likely:
        print("  (none found)")

    # Definitely literal
    print("\n❌ DEFINITELY LITERAL (score < 0.3):")
    literal = [(mwe, info) for mwe, info in classified.items()
               if info['idiomaticity_score'] < 0.3]
    literal.sort(key=lambda x: x[1]['frequency'], reverse=True)

    for mwe, info in literal[:15]:
        print(f"  • {mwe:<40} (score: {info['idiomaticity_score']:.3f}, freq: {info['frequency']:3d})")

    # Save classified results
    output_file = PROCESSED_DATA_DIR / "spanish_mwes_classified.csv"

    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)

    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['mwe', 'frequency', 'length', 'type', 'idiomaticity_score', 'is_idiomatic', 'known_idiom']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for mwe, info in sorted(classified.items(),
                               key=lambda x: x[1]['idiomaticity_score'],
                               reverse=True):
            writer.writerow({
                'mwe': mwe,
                'frequency': info['frequency'],
                'length': info['length'],
                'type': info['type'],
                'idiomaticity_score': f"{info['idiomaticity_score']:.3f}",
                'is_idiomatic': info['is_idiomatic'],
                'known_idiom': info['known_idiom']
            })

    print(f"✓ Saved classified MWEs to: {output_file}")

    # Save idiomatic candidates only
    idiomatic_file = PROCESSED_DATA_DIR / "spanish_idioms_candidates.csv"

    idiomatic_only = [(mwe, info) for mwe, info in classified.items()
                      if info['is_idiomatic']]

    with open(idiomatic_file, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['mwe', 'frequency', 'idiomaticity_score', 'known_idiom']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for mwe, info in sorted(idiomatic_only,
                               key=lambda x: x[1]['idiomaticity_score'],
                               reverse=True):
            writer.writerow({
                'mwe': mwe,
                'frequency': info['frequency'],
                'idiomaticity_score': f"{info['idiomaticity_score']:.3f}",
                'known_idiom': info['known_idiom']
            })

    print(f"✓ Saved {len(idiomatic_only):,} idiomatic candidates to: {idiomatic_file}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total MWEs processed: {len(classified):,}")
    print(f"Idiomatic candidates: {idiomatic_count:,} ({idiomatic_count/len(classified)*100:.1f}%)")
    print(f"Known idioms found: {known_count}")
    print(f"\nThese idiomatic candidates are ready for semantic matching with English idioms!")

    return idiomatic_count


if __name__ == "__main__":
    try:
        count = test_idiom_classification()
        print(f"\n✓ SUCCESS! Classified {count:,} as idiomatic")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
