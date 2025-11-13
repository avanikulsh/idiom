"""
Download and extract idioms with idiomatic contexts from MAGPIE dataset.
Extracts idioms with their usage examples to preserve semantic meaning.
"""
import json
import requests
from pathlib import Path
from typing import Set, List, Dict
from collections import defaultdict
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from config import ENGLISH_IDIOMS_DIR
except ImportError:
    # Fallback if running from project root
    from python.config import ENGLISH_IDIOMS_DIR


def download_magpie_idioms(
    output_format: str = 'json',
    max_examples_per_idiom: int = 3,
    min_confidence: float = 0.6,
    idiomatic_only: bool = True
) -> Dict[str, List[Dict]]:
    """
    Download MAGPIE dataset and extract idioms with idiomatic contexts.

    Args:
        output_format: 'json', 'txt', or 'csv'
        max_examples_per_idiom: Maximum context examples to keep per idiom
        min_confidence: Minimum annotation confidence (0-1)
        idiomatic_only: Only extract idiomatic uses (not literal)

    Returns:
        Dictionary mapping idioms to their contextual examples
    """
    print("Downloading MAGPIE dataset (this may take a minute)...")
    print(f"Settings: max_examples={max_examples_per_idiom}, min_confidence={min_confidence}, idiomatic_only={idiomatic_only}")

    # URL for the unfiltered MAGPIE dataset
    url = "https://raw.githubusercontent.com/hslh/magpie-corpus/master/MAGPIE_unfiltered.jsonl"

    try:
        # Stream the download to avoid loading everything into memory
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Group examples by idiom
        idiom_contexts: Dict[str, List[Dict]] = defaultdict(list)
        unique_idioms: Set[str] = set()

        print("Extracting idioms with idiomatic contexts...")
        line_count = 0
        idiomatic_count = 0

        # Process line by line
        for line in response.iter_lines(decode_unicode=True):
            if line:
                line_count += 1
                if line_count % 10000 == 0:
                    print(f"  Processed {line_count} entries, found {len(unique_idioms)} unique idioms ({idiomatic_count} idiomatic uses)")

                try:
                    entry = json.loads(line)
                    idiom = entry.get('idiom', '').strip()
                    label = entry.get('label', '')
                    confidence = entry.get('confidence', 0)

                    # Filter by confidence and idiomatic label
                    if confidence < min_confidence:
                        continue

                    # Label 'i' = idiomatic, 'l' = literal, 'f' = figurative, 'o' = other
                    if idiomatic_only and label not in ['i', 'f']:
                        continue

                    if not idiom:
                        continue

                    # Track unique idioms
                    unique_idioms.add(idiom)

                    # Only add if we haven't reached max examples for this idiom
                    if len(idiom_contexts[idiom]) < max_examples_per_idiom:
                        # Extract context (the sentence containing the idiom)
                        context = entry.get('context', [])
                        # context[2] is the sentence with the idiom (middle of 5-element array)
                        sentence = context[2] if len(context) > 2 else ' '.join(filter(None, context))

                        idiom_contexts[idiom].append({
                            'sentence': sentence,
                            'confidence': confidence,
                            'label': label,
                            'genre': entry.get('genre', ''),
                        })
                        idiomatic_count += 1

                except json.JSONDecodeError:
                    continue

        print(f"\nTotal entries processed: {line_count}")
        print(f"Unique idioms found: {len(unique_idioms)}")
        print(f"Total idiomatic examples: {idiomatic_count}")

        # Prepare data for export
        idiom_data = []
        for idiom in sorted(unique_idioms):
            contexts = idiom_contexts[idiom]
            idiom_data.append({
                'idiom': idiom,
                'examples': contexts,
                'num_examples': len(contexts),
                'source': 'MAGPIE'
            })

        # Save to file
        # Ensure directory exists
        ENGLISH_IDIOMS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = ENGLISH_IDIOMS_DIR / f"magpie_idioms_with_context.{output_format}"
        print(f"\nSaving to: {output_path}")

        if output_format == 'txt':
            # Plain text format: idiom followed by example sentences
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in idiom_data:
                    f.write(f"IDIOM: {item['idiom']}\n")
                    for i, ex in enumerate(item['examples'], 1):
                        f.write(f"  Example {i}: {ex['sentence']}\n")
                    f.write('\n')
            print(f"\nSaved {len(idiom_data)} idioms with contexts to: {output_path}")

        elif output_format == 'json':
            # JSON format with full metadata
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(idiom_data, f, indent=2, ensure_ascii=False)
            print(f"\nSaved {len(idiom_data)} idioms with contexts to: {output_path}")

        elif output_format == 'csv':
            # CSV format - flattened (one row per example)
            import csv
            rows = []
            for item in idiom_data:
                for ex in item['examples']:
                    rows.append({
                        'idiom': item['idiom'],
                        'sentence': ex['sentence'],
                        'confidence': ex['confidence'],
                        'label': ex['label'],
                        'genre': ex['genre'],
                        'source': 'MAGPIE'
                    })

            with open(output_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['idiom', 'sentence', 'confidence', 'label', 'genre', 'source'])
                writer.writeheader()
                writer.writerows(rows)
            print(f"\nSaved {len(rows)} idiom examples to: {output_path}")

        return idiom_contexts

    except Exception as e:
        print(f"Error downloading MAGPIE dataset: {e}")
        return []


def preview_idioms_with_context(idiom_contexts: Dict[str, List[Dict]], n: int = 10):
    """Preview first n idioms with their contexts"""
    print(f"\n{'='*80}")
    print(f"Preview of first {n} idioms with example contexts:")
    print('='*80)

    for i, (idiom, contexts) in enumerate(list(idiom_contexts.items())[:n], 1):
        print(f"\n{i}. IDIOM: {idiom}")
        print(f"   ({len(contexts)} example{'s' if len(contexts) != 1 else ''})")
        for j, ctx in enumerate(contexts, 1):
            print(f"   Example {j}: {ctx['sentence']}")
            print(f"   [confidence: {ctx['confidence']:.2f}, genre: {ctx.get('genre', 'N/A')}]")

    print('='*80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Download MAGPIE idiom dataset with idiomatic contexts'
    )
    parser.add_argument(
        '--format',
        choices=['txt', 'json', 'csv'],
        default='json',
        help='Output format (default: json)'
    )
    parser.add_argument(
        '--max-examples',
        type=int,
        default=3,
        help='Maximum examples per idiom (default: 3)'
    )
    parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.6,
        help='Minimum annotation confidence 0-1 (default: 0.6)'
    )
    parser.add_argument(
        '--include-literal',
        action='store_true',
        help='Include literal uses (default: idiomatic only)'
    )
    parser.add_argument(
        '--preview',
        type=int,
        default=10,
        help='Number of idioms to preview (default: 10)'
    )

    args = parser.parse_args()

    # Download and extract idioms with contexts
    idiom_contexts = download_magpie_idioms(
        output_format=args.format,
        max_examples_per_idiom=args.max_examples,
        min_confidence=args.min_confidence,
        idiomatic_only=not args.include_literal
    )

    if idiom_contexts:
        preview_idioms_with_context(idiom_contexts, n=args.preview)
        print(f"\n✓ Successfully extracted {len(idiom_contexts)} unique idioms from MAGPIE dataset")
        print(f"  Each idiom has up to {args.max_examples} contextual examples")
        print(f"  This preserves semantic meaning for cross-lingual matching!")
    else:
        print("\n✗ Failed to download idioms")
