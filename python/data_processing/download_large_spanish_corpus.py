"""
Download larger Spanish subtitle corpus from OpenSubtitles.
Target: 100k-200k sentences for finding idiom contexts.
"""
import requests
import random
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import SPANISH_SUBTITLES

def download_large_spanish_corpus(target_sentences=150000):
    """Download larger Spanish corpus."""

    print("="*80)
    print("DOWNLOADING LARGE SPANISH SUBTITLE CORPUS")
    print("="*80)

    # Use raw sentences file from OPUS
    url = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/es.txt.gz"

    print(f"\nDownloading from: {url}")
    print(f"Target: {target_sentences:,} sentences")
    print("This may take 5-10 minutes...")

    import gzip
    import io

    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()

    # Get total size
    total_size = int(response.headers.get('content-length', 0))
    print(f"File size: {total_size / (1024*1024):.1f} MB")

    # Decompress and read line by line
    sentences = []

    print("\nDownloading and processing...")

    # Read compressed data
    compressed_data = io.BytesIO()
    downloaded = 0

    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            compressed_data.write(chunk)
            downloaded += len(chunk)
            if downloaded % (1024*1024) == 0:
                print(f"  Downloaded: {downloaded / (1024*1024):.1f} MB", end='\r')

    print(f"\n✓ Downloaded {downloaded / (1024*1024):.1f} MB")

    # Decompress
    print("\nDecompressing...")
    compressed_data.seek(0)

    with gzip.GzipFile(fileobj=compressed_data) as f:
        for i, line in enumerate(f):
            try:
                line = line.decode('utf-8').strip()

                # Filter: length > 20 chars, has letters
                if len(line) > 20 and any(c.isalpha() for c in line):
                    sentences.append(line)

                    if len(sentences) >= target_sentences:
                        break

                if (i + 1) % 50000 == 0:
                    print(f"  Processed {i+1:,} lines, kept {len(sentences):,} sentences", end='\r')

            except Exception as e:
                continue

    print(f"\n✓ Collected {len(sentences):,} sentences")

    # Sample if we got more than needed
    if len(sentences) > target_sentences:
        print(f"\nRandomly sampling {target_sentences:,} sentences...")
        sentences = random.sample(sentences, target_sentences)

    # Save
    output_file = SPANISH_SUBTITLES / "spanish_opus_large.txt"

    print(f"\nSaving to: {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            f.write(sentence + '\n')

    file_size = output_file.stat().st_size / (1024*1024)
    print(f"✓ Saved {len(sentences):,} sentences ({file_size:.1f} MB)")

    # Show samples
    print("\n" + "="*80)
    print("SAMPLE SENTENCES")
    print("="*80)

    for i, sent in enumerate(random.sample(sentences, min(10, len(sentences))), 1):
        print(f"{i:2d}. {sent[:100]}...")

    return len(sentences)

if __name__ == "__main__":
    try:
        count = download_large_spanish_corpus(target_sentences=150000)
        print(f"\n✓ SUCCESS! Downloaded {count:,} Spanish sentences")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
