"""
Find usage contexts for Spanish idioms in subtitle corpus.
"""
import sys
import os
import csv
import re
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

from config import SPANISH_SUBTITLES, PROCESSED_DATA_DIR

def normalize_text(text):
    """Normalize text for matching."""
    # Convert to lowercase and remove extra punctuation
    text = text.lower()
    text = re.sub(r'[¿?¡!]', '', text)
    return text.strip()

def find_idiom_in_context(idiom, text):
    """Check if idiom appears in text with flexible matching."""
    idiom_norm = normalize_text(idiom)
    text_norm = normalize_text(text)

    # Try exact match first
    if idiom_norm in text_norm:
        return True

    # Try with different punctuation
    idiom_words = idiom_norm.split()
    if len(idiom_words) <= 1:
        return False

    # Check if main content words appear in order
    pattern = r'\b' + r'\b.*\b'.join(re.escape(word) for word in idiom_words) + r'\b'
    return bool(re.search(pattern, text_norm))

def find_contexts_for_idioms():
    """Find usage contexts for Spanish idioms in subtitle corpus."""

    print("="*80)
    print("FINDING SPANISH IDIOM CONTEXTS IN SUBTITLE CORPUS")
    print("="*80)

    # Load Spanish idioms
    from config import ENGLISH_IDIOMS_DIR
    spanish_file = ENGLISH_IDIOMS_DIR / "spanish_idioms_gavilan2021.csv"
    print(f"\nLoading Spanish idioms from: {spanish_file}")

    import pandas as pd
    df = pd.read_csv(spanish_file)
    idioms = df['idiom'].tolist()
    print(f"✓ Loaded {len(idioms)} Spanish idioms to search for")

    # Load Spanish subtitle corpus
    subtitle_file = SPANISH_SUBTITLES / "spanish_opus_10k_random.txt"
    print(f"\nLoading subtitle corpus from: {subtitle_file}")

    with open(subtitle_file, 'r', encoding='utf-8') as f:
        subtitles = [line.strip() for line in f if line.strip() and len(line.strip()) > 10]

    print(f"✓ Loaded {len(subtitles):,} subtitle lines")

    # Search for idiom contexts
    print("\n" + "="*80)
    print("SEARCHING FOR IDIOM OCCURRENCES")
    print("="*80)
    print("\nThis may take a few minutes...")

    idiom_contexts = defaultdict(list)

    for i, idiom in enumerate(idioms):
        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{len(idioms)} idioms...")

        for subtitle in subtitles:
            if find_idiom_in_context(idiom, subtitle):
                idiom_contexts[idiom].append(subtitle)

    # Statistics
    found_count = sum(1 for contexts in idiom_contexts.values() if contexts)
    total_contexts = sum(len(contexts) for contexts in idiom_contexts.values())

    print(f"\n✓ Search complete")
    print(f"  Idioms with contexts found: {found_count} / {len(idioms)} ({found_count/len(idioms)*100:.1f}%)")
    print(f"  Total contexts found: {total_contexts:,}")

    if found_count > 0:
        avg_contexts = total_contexts / found_count
        print(f"  Average contexts per idiom: {avg_contexts:.1f}")

    # Show examples
    print("\n" + "="*80)
    print("EXAMPLES OF FOUND CONTEXTS")
    print("="*80)

    found_idioms = [(idiom, contexts) for idiom, contexts in idiom_contexts.items() if contexts]
    found_idioms.sort(key=lambda x: len(x[1]), reverse=True)

    for idiom, contexts in found_idioms[:10]:
        print(f"\n{idiom} ({len(contexts)} occurrences)")
        for ctx in contexts[:2]:
            print(f"  • {ctx[:100]}...")

    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    output_file = PROCESSED_DATA_DIR / "spanish_idioms_with_contexts.csv"

    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['idiom', 'num_contexts', 'contexts'])

        for idiom in idioms:
            contexts = idiom_contexts[idiom]
            num_contexts = len(contexts)
            contexts_str = ' ||| '.join(contexts) if contexts else ''
            writer.writerow([idiom, num_contexts, contexts_str])

    print(f"✓ Saved results to: {output_file}")

    # Analysis
    print("\n" + "="*80)
    print("COVERAGE ANALYSIS")
    print("="*80)

    print(f"\nTotal idioms: {len(idioms)}")
    print(f"Idioms found in corpus: {found_count} ({found_count/len(idioms)*100:.1f}%)")
    print(f"Idioms NOT found: {len(idioms) - found_count} ({(len(idioms)-found_count)/len(idioms)*100:.1f}%)")

    # Distribution
    no_context = sum(1 for contexts in idiom_contexts.values() if len(contexts) == 0)
    one_context = sum(1 for contexts in idiom_contexts.values() if len(contexts) == 1)
    few_contexts = sum(1 for contexts in idiom_contexts.values() if 2 <= len(contexts) <= 5)
    many_contexts = sum(1 for contexts in idiom_contexts.values() if len(contexts) > 5)

    print(f"\nContext distribution:")
    print(f"  0 contexts: {no_context}")
    print(f"  1 context: {one_context}")
    print(f"  2-5 contexts: {few_contexts}")
    print(f"  >5 contexts: {many_contexts}")

    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)

    if found_count < len(idioms) * 0.3:
        print("\n⚠ WARNING: Very low coverage!")
        print("\nThe 10k subtitle sample is too small to find most idioms.")
        print("Recommendations:")
        print("  1. Download larger Spanish subtitle corpus (100k+ sentences)")
        print("  2. Use OpenSubtitles full corpus")
        print("  3. Search for Spanish idiom corpora with pre-annotated contexts")
        print("  4. Use web scraping to find idiom usage examples")
    else:
        print(f"\n✓ Found contexts for {found_count/len(idioms)*100:.1f}% of idioms")
        print("This may be sufficient for analysis, but more data would be better.")

    return found_count

if __name__ == "__main__":
    try:
        count = find_contexts_for_idioms()
        print(f"\n✓ SUCCESS! Found contexts for {count} Spanish idioms")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
