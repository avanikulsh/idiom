"""
Extract Spanish idioms with contexts from ID10M dataset.
Then match with Gavilán idioms to create a symmetric dataset.
"""
import csv
from pathlib import Path
from collections import defaultdict

def parse_bio_file(bio_file):
    """Parse BIO format file and extract sentences with idioms."""
    sentences_with_idioms = []
    current_sentence = []
    current_tags = []
    current_idiom_tokens = []
    in_idiom = False

    with open(bio_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if not line:  # Empty line = end of sentence
                if current_sentence and any(tag.startswith('B-') for tag in current_tags):
                    # This sentence has idioms
                    sentence_text = ' '.join(current_sentence)

                    # Extract idioms from this sentence
                    idioms_in_sent = []
                    idiom_tokens = []

                    for i, (token, tag) in enumerate(zip(current_sentence, current_tags)):
                        if tag == 'B-IDIOM':
                            if idiom_tokens:  # Save previous idiom
                                idioms_in_sent.append(' '.join(idiom_tokens))
                            idiom_tokens = [token]
                        elif tag == 'I-IDIOM' and idiom_tokens:
                            idiom_tokens.append(token)
                        elif tag == 'O' and idiom_tokens:
                            idioms_in_sent.append(' '.join(idiom_tokens))
                            idiom_tokens = []

                    if idiom_tokens:  # Don't forget last idiom
                        idioms_in_sent.append(' '.join(idiom_tokens))

                    sentences_with_idioms.append({
                        'sentence': sentence_text,
                        'idioms': idioms_in_sent
                    })

                # Reset for next sentence
                current_sentence = []
                current_tags = []
            else:
                parts = line.split('\t')
                if len(parts) == 2:
                    token, tag = parts
                    current_sentence.append(token)
                    current_tags.append(tag)

    return sentences_with_idioms

def group_by_idiom(sentences_with_idioms):
    """Group sentences by idiom."""
    idiom_contexts = defaultdict(list)

    for item in sentences_with_idioms:
        sentence = item['sentence']
        for idiom in item['idioms']:
            idiom_contexts[idiom].append(sentence)

    return idiom_contexts

def main():
    print("="*80)
    print("EXTRACTING SPANISH IDIOMS WITH CONTEXTS FROM ID10M")
    print("="*80)

    # Load ID10M data
    bio_file = Path("/Users/avani/Desktop/idiom-proj/data/raw/data/raw/data/raw/id10m_spanish/train_spanish.tsv")
    print(f"\nParsing BIO file: {bio_file}")

    sentences = parse_bio_file(bio_file)
    print(f"✓ Found {len(sentences)} sentences with idioms")

    # Group by idiom
    print("\nGrouping sentences by idiom...")
    idiom_contexts = group_by_idiom(sentences)
    print(f"✓ Found {len(idiom_contexts)} unique idioms")

    # Show top idioms
    print("\n" + "="*80)
    print("TOP 20 IDIOMS BY FREQUENCY")
    print("="*80)

    sorted_idioms = sorted(idiom_contexts.items(), key=lambda x: len(x[1]), reverse=True)

    for i, (idiom, contexts) in enumerate(sorted_idioms[:20], 1):
        print(f"{i:2d}. {idiom:40s} ({len(contexts)} contexts)")
        print(f"    Example: {contexts[0][:80]}...")

    # Save ID10M idioms with contexts
    output_file = Path("data/processed/id10m_spanish_idioms_with_contexts.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("SAVING ID10M IDIOMS WITH CONTEXTS")
    print("="*80)

    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['idiom', 'num_contexts', 'contexts'])

        for idiom, contexts in sorted_idioms:
            contexts_str = ' ||| '.join(contexts[:10])  # Save up to 10 contexts
            writer.writerow([idiom, len(contexts), contexts_str])

    print(f"✓ Saved {len(idiom_contexts)} idioms to: {output_file}")

    # Now load Gavilán idioms and find matches
    print("\n" + "="*80)
    print("MATCHING WITH GAVILÁN IDIOMS")
    print("="*80)

    gavilan_file = Path("/Users/avani/Desktop/idiom-proj/data/raw/english_idioms/spanish_idioms_gavilan2021.csv")
    print(f"\nLoading Gavilán idioms from: {gavilan_file}")

    gavilan_idioms = {}
    with open(gavilan_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gavilan_idioms[row['idiom']] = row

    print(f"✓ Loaded {len(gavilan_idioms)} Gavilán idioms")

    # Find matches
    matches = []
    partial_matches = []

    for gavilan_idiom in gavilan_idioms.keys():
        # Exact match
        if gavilan_idiom in idiom_contexts:
            matches.append((gavilan_idiom, idiom_contexts[gavilan_idiom]))
        else:
            # Partial match - check if Gavilán idiom is substring or vice versa
            for id10m_idiom, contexts in idiom_contexts.items():
                if gavilan_idiom.lower() in id10m_idiom.lower() or id10m_idiom.lower() in gavilan_idiom.lower():
                    partial_matches.append((gavilan_idiom, id10m_idiom, contexts))
                    break

    print(f"\n✓ Exact matches: {len(matches)}")
    print(f"✓ Partial matches: {len(partial_matches)}")

    # Save matched idioms with contexts
    matched_file = Path("data/processed/gavilan_idioms_with_id10m_contexts.csv")

    print("\n" + "="*80)
    print("SAVING GAVILÁN IDIOMS WITH ID10M CONTEXTS")
    print("="*80)

    with open(matched_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['gavilan_idiom', 'id10m_idiom', 'match_type', 'num_contexts', 'contexts',
                        'idiomatic_meaning', 'meaning_translation'])

        # Write exact matches
        for gavilan_idiom, contexts in matches:
            gav_data = gavilan_idioms[gavilan_idiom]
            contexts_str = ' ||| '.join(contexts[:5])  # Save up to 5 contexts
            writer.writerow([
                gavilan_idiom,
                gavilan_idiom,
                'exact',
                len(contexts),
                contexts_str,
                gav_data['idiomatic_meaning'],
                gav_data['idiomatic_meaning_translation']
            ])

        # Write partial matches
        for gavilan_idiom, id10m_idiom, contexts in partial_matches:
            gav_data = gavilan_idioms[gavilan_idiom]
            contexts_str = ' ||| '.join(contexts[:5])
            writer.writerow([
                gavilan_idiom,
                id10m_idiom,
                'partial',
                len(contexts),
                contexts_str,
                gav_data['idiomatic_meaning'],
                gav_data['idiomatic_meaning_translation']
            ])

    print(f"✓ Saved {len(matches) + len(partial_matches)} matched idioms to: {matched_file}")

    # Show examples
    print("\n" + "="*80)
    print("EXAMPLE MATCHES")
    print("="*80)

    for gav_idiom, contexts in matches[:10]:
        print(f"\n{gav_idiom}")
        print(f"  Contexts found: {len(contexts)}")
        print(f"  Example: {contexts[0][:100]}...")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total ID10M Spanish idioms: {len(idiom_contexts)}")
    print(f"Total Gavilán idioms: {len(gavilan_idioms)}")
    print(f"Matched with contexts: {len(matches) + len(partial_matches)} ({(len(matches)+len(partial_matches))/len(gavilan_idioms)*100:.1f}%)")

    return len(matches) + len(partial_matches)

if __name__ == "__main__":
    try:
        count = main()
        print(f"\n✓ SUCCESS! Matched {count} Spanish idioms with usage contexts")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
