"""
Create embeddings for all languages: English, French, Finnish, Japanese.
All use symmetric representation (idiom + usage contexts).
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


def create_idiom_representation(idiom, contexts, max_contexts=3):
    """Create text representation: idiom + contexts."""
    context_sample = contexts[:max_contexts]
    return f"{idiom}. " + " ".join(context_sample)


def create_embeddings_for_language(idioms, lang_name, model):
    """Create embeddings for a language."""
    print(f"\nCreating {lang_name} representations...")
    texts = [create_idiom_representation(item['idiom'], item['contexts'])
             for item in idioms]

    print(f"Sample {lang_name} representation:\n  {texts[0][:150]}...\n")

    print(f"Encoding {lang_name} idioms...")
    embeddings = model.encode(texts, show_progress_bar=True)
    print(f"✓ Encoded {len(embeddings):,} {lang_name} idioms")
    print(f"  Embedding shape: {embeddings.shape}")

    return embeddings, texts


def main():
    print("=" * 80)
    print("CREATING EMBEDDINGS FOR ALL LANGUAGES")
    print("=" * 80)
    print("\nSymmetric Representation: All languages use idiom + usage contexts")
    print("Languages: English, French, Finnish, Japanese\n")

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

    # Create embeddings
    print("\n" + "=" * 80)
    print("CREATING EMBEDDINGS")
    print("=" * 80)

    en_embeddings, en_texts = create_embeddings_for_language(
        english_idioms, "English", model
    )

    fr_embeddings, fr_texts = create_embeddings_for_language(
        french_idioms, "French", model
    )

    fi_embeddings, fi_texts = create_embeddings_for_language(
        finnish_idioms, "Finnish", model
    )

    jp_embeddings, jp_texts = create_embeddings_for_language(
        japanese_idioms, "Japanese", model
    )

    # Save embeddings
    print("\n" + "=" * 80)
    print("SAVING EMBEDDINGS")
    print("=" * 80)

    output_dir = Path("data/embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)

    # English (already saved, but update if needed)
    en_emb_file = output_dir / "english_idiom_embeddings.pkl"
    with open(en_emb_file, 'wb') as f:
        pickle.dump({
            'idioms': english_idioms,
            'embeddings': en_embeddings,
            'texts': en_texts
        }, f)
    print(f"\n✓ Saved English embeddings to: {en_emb_file}")

    # French
    fr_emb_file = output_dir / "french_idiom_embeddings.pkl"
    with open(fr_emb_file, 'wb') as f:
        pickle.dump({
            'idioms': french_idioms,
            'embeddings': fr_embeddings,
            'texts': fr_texts
        }, f)
    print(f"✓ Saved French embeddings to: {fr_emb_file}")

    # Finnish
    fi_emb_file = output_dir / "finnish_idiom_embeddings.pkl"
    with open(fi_emb_file, 'wb') as f:
        pickle.dump({
            'idioms': finnish_idioms,
            'embeddings': fi_embeddings,
            'texts': fi_texts
        }, f)
    print(f"✓ Saved Finnish embeddings to: {fi_emb_file}")

    # Japanese
    jp_emb_file = output_dir / "japanese_idiom_embeddings.pkl"
    with open(jp_emb_file, 'wb') as f:
        pickle.dump({
            'idioms': japanese_idioms,
            'embeddings': jp_embeddings,
            'texts': jp_texts
        }, f)
    print(f"✓ Saved Japanese embeddings to: {jp_emb_file}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\nEnglish:  {len(english_idioms):,} idioms ({en_embeddings.shape[1]} dims)")
    print(f"French:   {len(french_idioms):,} idioms ({fr_embeddings.shape[1]} dims)")
    print(f"Finnish:  {len(finnish_idioms):,} idioms ({fi_embeddings.shape[1]} dims)")
    print(f"Japanese: {len(japanese_idioms):,} idioms ({jp_embeddings.shape[1]} dims)")

    print(f"\nTotal: {len(english_idioms) + len(french_idioms) + len(finnish_idioms) + len(japanese_idioms):,} idioms across 4 languages")

    print("\nEmbeddings ready for:")
    print("  ✓ Cross-lingual semantic comparison")
    print("  ✓ Within-language similarity analysis")
    print("  ✓ Multilingual idiom clustering")

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
        print(f"\n✓ SUCCESS! Created embeddings for {total:,} idioms across 4 languages")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
