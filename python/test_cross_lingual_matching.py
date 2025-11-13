"""
Cross-lingual semantic matching of English and Spanish idioms.
Uses multilingual embeddings to find semantically similar idioms across languages.

Datasets:
- English: MAGPIE (Haagsma et al., 2020) - 1,730 idioms with contexts
- Spanish: Gavilán et al. (2021) - 1,252 idioms with meanings
"""
import sys
import os
import json
import csv
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from similarity.semantic_matcher import SemanticMatcher
from data_processing.idiom_loader import IdiomLoader
from config import ENGLISH_IDIOMS_DIR, RESULTS_DIR
import pandas as pd

def load_spanish_idioms(file_path):
    """Load Spanish idioms from Gavilán et al. dataset."""
    df = pd.read_csv(file_path)

    idioms = []
    for _, row in df.iterrows():
        idiom_data = {
            'text': row['idiom'],
            'idiomatic_meaning': row['idiomatic_meaning'],
            'meaning_translation': row['idiomatic_meaning_translation'],
            'literal_translation': row['idiom_literal_translation'],
            'familiarity': row['familiarity'],
            'source': 'Gavilán et al. (2021)'
        }
        idioms.append(idiom_data)

    return idioms


def test_cross_lingual_matching():
    """Test cross-lingual semantic matching between English and Spanish idioms."""

    print("="*80)
    print("CROSS-LINGUAL IDIOM SEMANTIC MATCHING")
    print("="*80)
    print("\nMatching English idioms (MAGPIE) with Spanish idioms (Gavilán et al. 2021)")
    print("Using multilingual sentence transformers for semantic similarity")

    # Load English idioms from MAGPIE
    print("\n" + "="*80)
    print("LOADING ENGLISH IDIOMS (MAGPIE)")
    print("="*80)

    magpie_file = ENGLISH_IDIOMS_DIR / "magpie_idioms_with_context.json"
    print(f"Loading from: {magpie_file}")

    # Load ONLY the MAGPIE English idioms (not the entire directory which includes Spanish)
    with open(magpie_file, 'r', encoding='utf-8') as f:
        magpie_data = json.load(f)

    english_idioms = []
    for item in magpie_data:
        idiom_entry = {
            'text': item['idiom'],
            'contexts': [ex.get('sentence', '') for ex in item.get('examples', [])],
            'source': 'MAGPIE'
        }
        english_idioms.append(idiom_entry)

    print(f"✓ Loaded {len(english_idioms)} English idioms with contexts")

    # Load Spanish idioms from Gavilán et al.
    print("\n" + "="*80)
    print("LOADING SPANISH IDIOMS (Gavilán et al. 2021)")
    print("="*80)

    spanish_file = ENGLISH_IDIOMS_DIR / "spanish_idioms_gavilan2021.csv"
    print(f"Loading from: {spanish_file}")

    spanish_idioms = load_spanish_idioms(spanish_file)
    print(f"✓ Loaded {len(spanish_idioms)} Spanish idioms")

    # Show samples
    print("\nEnglish idiom samples (with contexts):")
    for idiom in english_idioms[:3]:
        print(f"  • {idiom['text']}")
        if 'contexts' in idiom and idiom['contexts']:
            print(f"    Context: {idiom['contexts'][0][:80]}...")

    print("\nSpanish idiom samples (with meanings):")
    for idiom in spanish_idioms[:3]:
        print(f"  • {idiom['text']}")
        print(f"    Meaning: {idiom['meaning_translation'][:80]}...")

    # Initialize semantic matcher
    print("\n" + "="*80)
    print("INITIALIZING SEMANTIC MATCHER")
    print("="*80)

    print("Loading multilingual sentence transformer model...")
    print("This may take a minute on first run...")

    matcher = SemanticMatcher()
    print("✓ Model loaded")

    # Prepare data for encoding
    print("\n" + "="*80)
    print("ENCODING IDIOMS WITH SEMANTIC CONTEXT")
    print("="*80)

    print("\nApproach:")
    print("  • English: Encode idiom + usage contexts to capture semantic meaning")
    print("    Example: 'break the ice: The question helped break the ice at...'")
    print("  • Spanish: Encode idiom + idiomatic meaning explanation")
    print("    Example: 'romper el hielo: iniciar una conversación para relajar...'")
    print("  • Then: Find cross-lingual pairs with similar semantic embeddings")

    print(f"\nEncoding {len(english_idioms)} English idioms (idiom + contexts)...")

    # For Spanish: combine idiom with its idiomatic meaning for semantic representation
    print(f"Encoding {len(spanish_idioms)} Spanish idioms (idiom + meanings)...")
    spanish_texts = [
        f"{idiom['text']}: {idiom['idiomatic_meaning']}"
        for idiom in spanish_idioms
    ]

    # Find semantic matches
    print("\n" + "="*80)
    print("COMPUTING SEMANTIC SIMILARITY")
    print("="*80)

    print("Finding cross-lingual semantic matches...")
    print("This may take a few minutes...")

    matches = matcher.find_similar_mwes(
        english_idioms=english_idioms,  # Passes dicts with contexts
        foreign_mwes=spanish_texts,
        threshold=0.5,  # Lower threshold for cross-lingual
        top_k=5
    )

    print(f"\n✓ Found matches for {len(matches)} English idioms")

    # Analyze results
    print("\n" + "="*80)
    print("TOP 30 CROSS-LINGUAL MATCHES")
    print("="*80)

    # Convert to list and sort by best match score
    match_list = []
    for eng_idiom, spanish_matches in matches.items():
        if spanish_matches:
            best_score = spanish_matches[0][1]
            match_list.append((eng_idiom, spanish_matches, best_score))

    match_list.sort(key=lambda x: x[2], reverse=True)

    print(f"\n{'Rank':<5} {'English Idiom':<40} {'Score':<7} {'Spanish Match'}")
    print("-"*80)

    for i, (eng_idiom, sp_matches, best_score) in enumerate(match_list[:30], 1):
        # Extract just the idiom text from Spanish match
        sp_text = sp_matches[0][0].split(':')[0].strip()
        print(f"{i:<5} {eng_idiom[:38]:<40} {best_score:.3f}   {sp_text[:35]}")

    # Show detailed examples
    print("\n" + "="*80)
    print("DETAILED MATCH EXAMPLES (Top 10)")
    print("="*80)

    for i, (eng_idiom, sp_matches, _) in enumerate(match_list[:10], 1):
        print(f"\n{i}. ENGLISH: {eng_idiom}")

        # Find the English idiom data
        eng_data = next((item for item in english_idioms if item['text'] == eng_idiom), None)
        if eng_data and 'contexts' in eng_data and eng_data['contexts']:
            print(f"   Context: {eng_data['contexts'][0][:100]}...")

        print(f"\n   SPANISH MATCHES:")
        for j, (sp_match, score) in enumerate(sp_matches, 1):
            # Parse Spanish match
            sp_parts = sp_match.split(':', 1)
            sp_idiom = sp_parts[0].strip()

            # Find Spanish data
            sp_data = next((item for item in spanish_idioms if item['text'] == sp_idiom), None)

            print(f"   {j}. {sp_idiom} (similarity: {score:.3f})")
            if sp_data:
                print(f"      Meaning: {sp_data['meaning_translation'][:80]}...")

    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save as CSV
    csv_file = RESULTS_DIR / "english_spanish_idiom_matches.csv"
    with open(csv_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['english_idiom', 'spanish_idiom', 'similarity_score',
                        'spanish_meaning_en', 'rank'])

        for eng_idiom, sp_matches, _ in match_list:
            for rank, (sp_match, score) in enumerate(sp_matches, 1):
                sp_idiom = sp_match.split(':')[0].strip()
                sp_data = next((item for item in spanish_idioms if item['text'] == sp_idiom), None)
                meaning = sp_data['meaning_translation'] if sp_data else ''

                writer.writerow([eng_idiom, sp_idiom, f"{score:.4f}", meaning, rank])

    print(f"✓ Saved CSV to: {csv_file}")

    # Save as JSON
    json_file = RESULTS_DIR / "english_spanish_idiom_matches.json"
    json_data = {
        'metadata': {
            'english_dataset': 'MAGPIE (Haagsma et al., 2020)',
            'spanish_dataset': 'Gavilán et al. (2021)',
            'total_english_idioms': len(english_idioms),
            'total_spanish_idioms': len(spanish_idioms),
            'matches_found': len(matches),
            'similarity_threshold': 0.5,
            'model': 'paraphrase-multilingual-mpnet-base-v2'
        },
        'matches': []
    }

    for eng_idiom, sp_matches, best_score in match_list:
        eng_data = next((item for item in english_idioms if item['text'] == eng_idiom), None)

        match_entry = {
            'english_idiom': eng_idiom,
            'english_context': eng_data['contexts'][0] if eng_data and 'contexts' in eng_data and eng_data['contexts'] else '',
            'best_match_score': best_score,
            'spanish_matches': []
        }

        for sp_match, score in sp_matches:
            sp_idiom = sp_match.split(':')[0].strip()
            sp_data = next((item for item in spanish_idioms if item['text'] == sp_idiom), None)

            match_entry['spanish_matches'].append({
                'idiom': sp_idiom,
                'similarity_score': score,
                'meaning_spanish': sp_data['idiomatic_meaning'] if sp_data else '',
                'meaning_english': sp_data['meaning_translation'] if sp_data else '',
                'literal_translation': sp_data['literal_translation'] if sp_data else ''
            })

        json_data['matches'].append(match_entry)

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved JSON to: {json_file}")

    # Statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)

    scores = [score for _, sp_matches, _ in match_list for _, score in sp_matches]

    print(f"\nTotal English idioms: {len(english_idioms)}")
    print(f"Total Spanish idioms: {len(spanish_idioms)}")
    print(f"English idioms with matches: {len(matches)} ({len(matches)/len(english_idioms)*100:.1f}%)")
    print(f"Total match pairs: {len(scores)}")

    if scores:
        print(f"\nSimilarity scores:")
        print(f"  Mean: {sum(scores)/len(scores):.3f}")
        print(f"  Max: {max(scores):.3f}")
        print(f"  Min: {min(scores):.3f}")

        # Distribution
        high = sum(1 for s in scores if s >= 0.7)
        medium = sum(1 for s in scores if 0.6 <= s < 0.7)
        low = sum(1 for s in scores if s < 0.6)

        print(f"\nScore distribution:")
        print(f"  High (≥0.7): {high} ({high/len(scores)*100:.1f}%)")
        print(f"  Medium (0.6-0.7): {medium} ({medium/len(scores)*100:.1f}%)")
        print(f"  Low (<0.6): {low} ({low/len(scores)*100:.1f}%)")

    print("\n" + "="*80)
    print("✓ CROSS-LINGUAL MATCHING COMPLETE")
    print("="*80)
    print("\nResults ready for analysis and publication!")

    return len(matches)


if __name__ == "__main__":
    try:
        count = test_cross_lingual_matching()
        print(f"\n✓ SUCCESS! Found matches for {count} English idioms")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
