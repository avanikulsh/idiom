"""
Extract Finnish and Japanese idioms with contexts from Crossing the Threshold dataset.
"""
import csv
from pathlib import Path
from collections import defaultdict

def extract_language_idioms(lang_code, lang_name):
    """Extract idioms for a specific language."""
    print("=" * 80)
    print(f"EXTRACTING {lang_name.upper()} IDIOMS WITH CONTEXTS")
    print("=" * 80)

    idiom_file = Path(f"data/raw/idiom-translation/metaphor-translation/data/test_sets_final/{lang_code}/idiomatic_all_fixed.csv")

    print(f"\nLoading from: {idiom_file}")

    with open(idiom_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)

    print(f"✓ Loaded {len(data):,} idiom contexts")

    # Group by idiom
    idiom_contexts = defaultdict(list)

    for row in data:
        idiom = row['contains_idioms']
        context = row['original_text']
        english_translation = row['text']

        idiom_contexts[idiom].append({
            f'{lang_code}_context': context,
            'english_translation': english_translation
        })

    print(f"✓ Found {len(idiom_contexts):,} unique {lang_name} idioms")

    # Show top idioms
    print(f"\n{'=' * 80}")
    print(f"TOP 20 {lang_name.upper()} IDIOMS BY FREQUENCY")
    print("=" * 80)

    sorted_idioms = sorted(idiom_contexts.items(), key=lambda x: len(x[1]), reverse=True)

    for i, (idiom, contexts) in enumerate(sorted_idioms[:20], 1):
        print(f"\n{i:2d}. {idiom:50s} ({len(contexts)} contexts)")
        print(f"    {lang_name}: {contexts[0][f'{lang_code}_context'][:70]}...")
        print(f"    English: {contexts[0]['english_translation'][:70]}...")

    # Save to CSV
    output_file = Path(f"data/processed/{lang_code}_idioms_with_contexts.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 80}")
    print(f"SAVING {lang_name.upper()} IDIOMS")
    print("=" * 80)

    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['idiom', 'num_contexts', f'{lang_code}_contexts', 'english_translations'])

        for idiom, contexts in sorted_idioms:
            lang_contexts_str = ' ||| '.join([c[f'{lang_code}_context'] for c in contexts[:5]])
            english_translations_str = ' ||| '.join([c['english_translation'] for c in contexts[:5]])
            writer.writerow([idiom, len(contexts), lang_contexts_str, english_translations_str])

    print(f"\n✓ Saved {len(idiom_contexts):,} {lang_name} idioms to: {output_file}")

    return len(idiom_contexts), sorted_idioms

def main():
    print("=" * 80)
    print("EXTRACTING FINNISH AND JAPANESE IDIOMS")
    print("=" * 80)
    print("\nFrom: Crossing the Threshold (2023) dataset")
    print("Format: idiom + usage contexts from movie subtitles\n")

    # Extract Finnish
    fi_count, fi_idioms = extract_language_idioms('fi', 'Finnish')

    print("\n")

    # Extract Japanese
    jp_count, jp_idioms = extract_language_idioms('jp', 'Japanese')

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\nFinnish: {fi_count:,} unique idioms")
    print(f"Japanese: {jp_count:,} unique idioms")
    print(f"Total: {fi_count + jp_count:,} idioms extracted")

    print("\nAll idioms have:")
    print("  ✓ Original language contexts")
    print("  ✓ English translations")
    print("  ✓ Symmetric representation (idiom + contexts)")

    return fi_count, jp_count

if __name__ == "__main__":
    try:
        fi_count, jp_count = main()
        print(f"\n✓ SUCCESS! Extracted {fi_count:,} Finnish and {jp_count:,} Japanese idioms")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
