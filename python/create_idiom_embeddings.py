"""
Create embeddings for English and French idioms for semantic comparison.
Both languages use symmetric representations: idiom + usage contexts.

This script creates embeddings without assuming 1-to-1 matching.
Instead, it enables exploratory semantic analysis within and across languages.
"""
import json
import csv
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_english_idioms(magpie_file):
    """Load English idioms from MAGPIE with contexts."""
    print(f"\nLoading English idioms from: {magpie_file}")

    with open(magpie_file, 'r', encoding='utf-8') as f:
        magpie_data = json.load(f)

    english_idioms = []
    for item in magpie_data:
        contexts = [ex.get('sentence', '') for ex in item.get('examples', [])]

        if contexts:  # Only include idioms with contexts
            english_idioms.append({
                'idiom': item['idiom'],
                'contexts': contexts,
                'source': 'MAGPIE'
            })

    print(f"✓ Loaded {len(english_idioms):,} English idioms with contexts")
    return english_idioms


def load_french_idioms(french_file):
    """Load French idioms with contexts."""
    print(f"\nLoading French idioms from: {french_file}")

    french_idioms = []

    with open(french_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            french_contexts = row['french_contexts'].split(' ||| ')

            french_idioms.append({
                'idiom': row['idiom'],
                'contexts': french_contexts,
                'num_contexts': int(row['num_contexts']),
                'english_translations': row['english_translations'].split(' ||| ')
            })

    print(f"✓ Loaded {len(french_idioms):,} French idioms with contexts")
    return french_idioms


def create_idiom_representation(idiom, contexts, max_contexts=3):
    """Create text representation: idiom + contexts."""
    context_sample = contexts[:max_contexts]
    return f"{idiom}. " + " ".join(context_sample)


def analyze_within_language_similarity(idioms, embeddings, lang_name, top_k=5):
    """Analyze semantic similarity within a language."""
    print(f"\n{'=' * 80}")
    print(f"WITHIN-{lang_name.upper()} SEMANTIC SIMILARITY")
    print("=" * 80)

    # Compute similarity matrix
    sim_matrix = cosine_similarity(embeddings)

    # For each idiom, find most similar idioms (excluding itself)
    examples = []

    for i in range(min(10, len(idioms))):
        # Get similarities, excluding self
        sims = sim_matrix[i].copy()
        sims[i] = -1  # Exclude self

        # Get top k similar idioms
        top_indices = np.argsort(sims)[-top_k:][::-1]

        print(f"\n{i+1}. {idioms[i]['idiom']}")
        print(f"   Context: {idioms[i]['contexts'][0][:80]}...")
        print(f"   Most similar {lang_name} idioms:")

        similar_idioms = []
        for rank, idx in enumerate(top_indices, 1):
            print(f"      {rank}. {idioms[idx]['idiom']:40s} (sim: {sims[idx]:.3f})")
            similar_idioms.append({
                'idiom': idioms[idx]['idiom'],
                'similarity': float(sims[idx])
            })

        examples.append({
            'idiom': idioms[i]['idiom'],
            'context': idioms[i]['contexts'][0],
            'similar_idioms': similar_idioms
        })

    return examples


def main():
    print("=" * 80)
    print("IDIOM EMBEDDINGS FOR SEMANTIC COMPARISON")
    print("=" * 80)
    print("\nSymmetric Representation: Both languages use idiom + usage contexts")
    print("English: MAGPIE idioms with BNC contexts")
    print("French: Crossing the Threshold idioms with movie subtitle contexts")

    # Load data
    magpie_file = Path("data/raw/english_idioms/magpie_idioms_with_context.json")
    french_file = Path("data/processed/french_idioms_with_contexts.csv")

    english_idioms = load_english_idioms(magpie_file)
    french_idioms = load_french_idioms(french_file)

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

    print("\nCreating English representations...")
    english_texts = [create_idiom_representation(item['idiom'], item['contexts'])
                     for item in english_idioms]

    print(f"Sample English representation:\n  {english_texts[0][:150]}...\n")

    print("Encoding English idioms...")
    english_embeddings = model.encode(english_texts, show_progress_bar=True)
    print(f"✓ Encoded {len(english_embeddings):,} English idioms")
    print(f"  Embedding shape: {english_embeddings.shape}")

    print("\nCreating French representations...")
    french_texts = [create_idiom_representation(item['idiom'], item['contexts'])
                    for item in french_idioms]

    print(f"Sample French representation:\n  {french_texts[0][:150]}...\n")

    print("Encoding French idioms...")
    french_embeddings = model.encode(french_texts, show_progress_bar=True)
    print(f"✓ Encoded {len(french_embeddings):,} French idioms")
    print(f"  Embedding shape: {french_embeddings.shape}")

    # Save embeddings
    print("\n" + "=" * 80)
    print("SAVING EMBEDDINGS")
    print("=" * 80)

    output_dir = Path("data/embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save English embeddings
    en_emb_file = output_dir / "english_idiom_embeddings.pkl"
    with open(en_emb_file, 'wb') as f:
        pickle.dump({
            'idioms': english_idioms,
            'embeddings': english_embeddings,
            'texts': english_texts
        }, f)
    print(f"\n✓ Saved English embeddings to: {en_emb_file}")

    # Save French embeddings
    fr_emb_file = output_dir / "french_idiom_embeddings.pkl"
    with open(fr_emb_file, 'wb') as f:
        pickle.dump({
            'idioms': french_idioms,
            'embeddings': french_embeddings,
            'texts': french_texts
        }, f)
    print(f"✓ Saved French embeddings to: {fr_emb_file}")

    # Analyze within-language semantic similarity
    en_examples = analyze_within_language_similarity(
        english_idioms, english_embeddings, "English", top_k=5
    )

    fr_examples = analyze_within_language_similarity(
        french_idioms, french_embeddings, "French", top_k=5
    )

    # Save similarity examples
    print("\n" + "=" * 80)
    print("SAVING SIMILARITY EXAMPLES")
    print("=" * 80)

    examples_file = output_dir / "within_language_similarity_examples.json"
    with open(examples_file, 'w', encoding='utf-8') as f:
        json.dump({
            'english_examples': en_examples,
            'french_examples': fr_examples
        }, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved similarity examples to: {examples_file}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\nEnglish idioms: {len(english_idioms):,}")
    print(f"  Embedding dimensions: {english_embeddings.shape[1]}")
    print(f"  Average contexts per idiom: {np.mean([len(i['contexts']) for i in english_idioms]):.1f}")

    print(f"\nFrench idioms: {len(french_idioms):,}")
    print(f"  Embedding dimensions: {french_embeddings.shape[1]}")
    print(f"  Average contexts per idiom: {np.mean([len(i['contexts']) for i in french_idioms]):.1f}")

    print("\nEmbeddings saved and ready for semantic comparison!")
    print("\nUse cases:")
    print("  - Within-language idiom similarity analysis")
    print("  - Cross-lingual semantic comparison")
    print("  - Clustering and semantic space exploration")
    print("  - Finding semantically related idioms")

    return len(english_idioms), len(french_idioms)


if __name__ == "__main__":
    try:
        en_count, fr_count = main()
        print(f"\n✓ SUCCESS! Created embeddings for {en_count:,} English and {fr_count:,} French idioms")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
