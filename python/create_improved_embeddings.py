"""
Create improved embeddings for more accurate cross-lingual idiom matching.
Uses dual representation: idiom-only + idiom-with-contexts.
"""
import json
import csv
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np


def load_english_idioms(magpie_file):
    """Load English idioms from MAGPIE with contexts."""
    print(f"Loading English idioms from: {magpie_file}")

    with open(magpie_file, 'r', encoding='utf-8') as f:
        magpie_data = json.load(f)

    english_idioms = []
    for item in magpie_data:
        contexts = [ex.get('sentence', '') for ex in item.get('examples', [])]

        if contexts:
            english_idioms.append({
                'idiom': item['idiom'],
                'contexts': contexts,
                'source': 'MAGPIE'
            })

    print(f"✓ Loaded {len(english_idioms):,} English idioms with contexts")
    return english_idioms


def load_target_language_idioms(csv_file, lang_code):
    """Load target language idioms with contexts."""
    print(f"Loading {lang_code.upper()} idioms from: {csv_file}")

    idioms = []

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Handle different column naming conventions
            if f'{lang_code}_contexts' in row:
                contexts = row[f'{lang_code}_contexts'].split(' ||| ')
            elif 'french_contexts' in row and lang_code == 'fr':
                contexts = row['french_contexts'].split(' ||| ')
            else:
                # Fallback: find any column with 'context' in name
                context_col = [col for col in row.keys() if 'context' in col.lower()][0]
                contexts = row[context_col].split(' ||| ')

            idioms.append({
                'idiom': row['idiom'],
                'contexts': contexts,
                'num_contexts': int(row['num_contexts']),
                'english_translations': row['english_translations'].split(' ||| ')
            })

    print(f"✓ Loaded {len(idioms):,} {lang_code.upper()} idioms with contexts")
    return idioms


def create_dual_representations(idiom, contexts, max_contexts=3):
    """
    Create two representations:
    1. Idiom only (for structural/metaphorical similarity)
    2. Idiom + contexts (for usage-based similarity)
    """
    # Idiom-only representation
    idiom_only = idiom

    # Idiom + contexts representation
    context_sample = contexts[:max_contexts]
    idiom_with_contexts = f"{idiom}. " + " ".join(context_sample)

    return idiom_only, idiom_with_contexts


def create_dual_embeddings_for_language(idioms, lang_name, model):
    """Create dual embeddings for a language."""
    print(f"\nCreating dual {lang_name} representations...")

    idiom_only_texts = []
    idiom_context_texts = []

    for item in idioms:
        idiom_only, idiom_context = create_dual_representations(
            item['idiom'],
            item['contexts']
        )
        idiom_only_texts.append(idiom_only)
        idiom_context_texts.append(idiom_context)

    print(f"Sample {lang_name} idiom-only: {idiom_only_texts[0]}")
    print(f"Sample {lang_name} with context: {idiom_context_texts[0][:150]}...")

    print(f"\nEncoding {lang_name} idioms (idiom-only)...")
    idiom_only_embeddings = model.encode(idiom_only_texts, show_progress_bar=True)
    print(f"✓ Encoded {len(idiom_only_embeddings):,} idiom-only embeddings")

    print(f"Encoding {lang_name} idioms (with contexts)...")
    idiom_context_embeddings = model.encode(idiom_context_texts, show_progress_bar=True)
    print(f"✓ Encoded {len(idiom_context_embeddings):,} idiom+context embeddings")

    return {
        'idiom_only': idiom_only_embeddings,
        'idiom_context': idiom_context_embeddings,
        'idiom_only_texts': idiom_only_texts,
        'idiom_context_texts': idiom_context_texts
    }


def main():
    print("=" * 80)
    print("CREATING IMPROVED DUAL EMBEDDINGS")
    print("=" * 80)
    print("\nDual Representation Strategy:")
    print("  1. Idiom-only: Captures metaphorical/structural similarity")
    print("  2. Idiom+context: Captures usage-based similarity")
    print("\nThis approach reduces lexical overlap bias and improves match quality.\n")

    # Load data
    magpie_file = Path("data/raw/english_idioms/magpie_idioms_with_context.json")
    fr_file = Path("data/processed/french_idioms_with_contexts.csv")
    fi_file = Path("data/processed/fi_idioms_with_contexts.csv")
    jp_file = Path("data/processed/jp_idioms_with_contexts.csv")

    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    print()

    english_idioms = load_english_idioms(magpie_file)
    french_idioms = load_target_language_idioms(fr_file, 'fr')
    finnish_idioms = load_target_language_idioms(fi_file, 'fi')
    japanese_idioms = load_target_language_idioms(jp_file, 'jp')

    # Load multilingual sentence transformer
    print("\n" + "=" * 80)
    print("LOADING MULTILINGUAL SENTENCE TRANSFORMER")
    print("=" * 80)
    print("\nModel: paraphrase-multilingual-mpnet-base-v2")

    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    print("✓ Model loaded")

    # Create dual embeddings
    print("\n" + "=" * 80)
    print("CREATING DUAL EMBEDDINGS")
    print("=" * 80)

    en_embeddings = create_dual_embeddings_for_language(
        english_idioms, "English", model
    )

    fr_embeddings = create_dual_embeddings_for_language(
        french_idioms, "French", model
    )

    fi_embeddings = create_dual_embeddings_for_language(
        finnish_idioms, "Finnish", model
    )

    jp_embeddings = create_dual_embeddings_for_language(
        japanese_idioms, "Japanese", model
    )

    # Save embeddings
    print("\n" + "=" * 80)
    print("SAVING DUAL EMBEDDINGS")
    print("=" * 80)

    output_dir = Path("data/embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)

    languages = [
        ('english', english_idioms, en_embeddings),
        ('french', french_idioms, fr_embeddings),
        ('finnish', finnish_idioms, fi_embeddings),
        ('japanese', japanese_idioms, jp_embeddings)
    ]

    for lang_name, idioms, embeddings in languages:
        emb_file = output_dir / f"{lang_name}_dual_embeddings.pkl"
        with open(emb_file, 'wb') as f:
            pickle.dump({
                'idioms': idioms,
                'idiom_only_embeddings': embeddings['idiom_only'],
                'idiom_context_embeddings': embeddings['idiom_context'],
                'idiom_only_texts': embeddings['idiom_only_texts'],
                'idiom_context_texts': embeddings['idiom_context_texts']
            }, f)
        print(f"✓ Saved {lang_name} dual embeddings to: {emb_file}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\nEnglish:  {len(english_idioms):,} idioms (2 × {en_embeddings['idiom_only'].shape[1]} dims)")
    print(f"French:   {len(french_idioms):,} idioms (2 × {fr_embeddings['idiom_only'].shape[1]} dims)")
    print(f"Finnish:  {len(finnish_idioms):,} idioms (2 × {fi_embeddings['idiom_only'].shape[1]} dims)")
    print(f"Japanese: {len(japanese_idioms):,} idioms (2 × {jp_embeddings['idiom_only'].shape[1]} dims)")

    total = len(english_idioms) + len(french_idioms) + len(finnish_idioms) + len(japanese_idioms)
    print(f"\nTotal: {total:,} idioms × 2 embedding types = {total * 2:,} embeddings")

    print("\nDual embeddings enable:")
    print("  ✓ Weighted scoring (idiom similarity + context similarity)")
    print("  ✓ Reduced lexical overlap bias")
    print("  ✓ Better metaphorical structure matching")
    print("  ✓ More accurate cross-lingual idiom matching")

    return {
        'en': len(english_idioms),
        'fr': len(french_idioms),
        'fi': len(finnish_idioms),
        'jp': len(japanese_idioms)
    }


if __name__ == "__main__":
    try:
        counts = main()
        total = sum(counts.values())
        print(f"\n✓ SUCCESS! Created dual embeddings for {total:,} idioms across 4 languages")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
