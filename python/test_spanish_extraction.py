"""
Test MWE extraction on Spanish subtitle data.
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from mwe_extraction.extractor import MWEExtractor
from config import SPANISH_SUBTITLES

def test_spanish_mwe_extraction():
    """Test MWE extraction on Spanish subtitle sample."""

    print("="*70)
    print("Testing Spanish MWE Extraction")
    print("="*70)

    # Load Spanish subtitle data
    spanish_file = SPANISH_SUBTITLES / "spanish_opus_10k_random.txt"

    print(f"\nLoading Spanish subtitles from: {spanish_file}")

    with open(spanish_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]

    print(f"✓ Loaded {len(texts):,} subtitle lines")

    # Initialize Spanish MWE extractor
    print("\nInitializing Spanish MWE extractor...")
    print("Note: Using multilingual spaCy model (xx_ent_wiki_sm)")
    print("For better results, install: python -m spacy download es_core_news_sm")

    try:
        extractor = MWEExtractor(language='es', spacy_model='es_core_news_sm')
        print("✓ Loaded Spanish spaCy model")
    except:
        print("⚠ Spanish model not found, using multilingual model")
        extractor = MWEExtractor(language='es', spacy_model='xx_ent_wiki_sm')

    # Extract MWEs
    print("\nExtracting candidate MWEs...")
    print("This may take a minute...")

    mwes = extractor.extract_candidate_mwes(
        texts=texts,
        min_length=2,
        max_length=6,
        min_freq=3  # Appear at least 3 times
    )

    print(f"\n✓ Extracted {len(mwes):,} candidate MWEs")

    # Convert to sorted list
    mwe_list = sorted(mwes.items(), key=lambda x: x[1]['frequency'], reverse=True)

    # Show statistics
    print("\n" + "="*70)
    print("EXTRACTION STATISTICS")
    print("="*70)

    by_type = {}
    by_length = {}

    for mwe, info in mwe_list:
        mwe_type = info['type']
        mwe_len = info['length']

        by_type[mwe_type] = by_type.get(mwe_type, 0) + 1
        by_length[mwe_len] = by_length.get(mwe_len, 0) + 1

    print(f"\nBy type:")
    for mwe_type, count in sorted(by_type.items(), key=lambda x: x[1], reverse=True):
        print(f"  {mwe_type}: {count:,}")

    print(f"\nBy length:")
    for length, count in sorted(by_length.items()):
        print(f"  {length} words: {count:,}")

    # Show top MWEs
    print("\n" + "="*70)
    print("TOP 30 MOST FREQUENT MWEs")
    print("="*70)

    for i, (mwe, info) in enumerate(mwe_list[:30], 1):
        print(f"{i:2d}. {mwe:40s} (freq: {info['frequency']:3d}, type: {info['type']:12s})")

    # Show samples by type
    print("\n" + "="*70)
    print("SAMPLES BY TYPE")
    print("="*70)

    for mwe_type in ['ngram', 'noun_phrase', 'verb_phrase']:
        print(f"\n{mwe_type.upper()}:")
        type_samples = [(mwe, info) for mwe, info in mwe_list if info['type'] == mwe_type]

        if type_samples:
            for i, (mwe, info) in enumerate(type_samples[:10], 1):
                print(f"  {i:2d}. {mwe:35s} (freq: {info['frequency']:3d})")
        else:
            print("  (none found)")

    # Save results
    output_file = SPANISH_SUBTITLES.parent.parent / "processed" / "spanish_mwes.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)

    import csv
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['mwe', 'frequency', 'length', 'type'])
        for mwe, info in mwe_list:
            writer.writerow([mwe, info['frequency'], info['length'], info['type']])

    print(f"✓ Saved {len(mwe_list):,} MWEs to: {output_file}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total MWEs extracted: {len(mwe_list):,}")
    print(f"From {len(texts):,} subtitle lines")
    print(f"Minimum frequency: 3 occurrences")
    print("\nThese Spanish MWEs are ready for semantic matching with English idioms!")

    return len(mwe_list)


if __name__ == "__main__":
    try:
        count = test_spanish_mwe_extraction()
        print(f"\n✓ SUCCESS! Extracted {count:,} Spanish MWEs")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
