"""
Download RANDOM samples of Spanish subtitles from OPUS.
Strategy: Download a moderate chunk, then randomly sample lines for diversity.
"""
import requests
import gzip
import os
import sys
import random
from pathlib import Path
import io

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from config import SPANISH_SUBTITLES
except ImportError:
    from python.config import SPANISH_SUBTITLES


def download_random_spanish_sample(
    chunk_size_mb: int = 20,
    sample_size: int = 10000,
    seed: int = 42
):
    """
    Download a chunk of OPUS Spanish data and randomly sample lines.

    Args:
        chunk_size_mb: How many MB to download (compressed) for sampling pool
        sample_size: Number of random lines to keep
        seed: Random seed for reproducibility

    Returns:
        Number of lines sampled
    """
    print(f"Strategy: Download {chunk_size_mb}MB chunk, randomly sample {sample_size:,} lines")
    print("This gives diverse data without downloading the full 2GB corpus\n")

    url = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/es.txt.gz"

    # MUST start from byte 0 for gzip decompression to work
    # We'll get diversity from random sampling the lines, not random file position
    random.seed(seed)
    chunk_size_bytes = chunk_size_mb * 1024 * 1024
    start_byte = 0
    end_byte = chunk_size_bytes - 1

    print(f"Downloading first {chunk_size_mb}MB from beginning of file")
    print(f"(Will randomly sample {sample_size:,} lines from this chunk)\n")

    try:
        # HTTP range request
        headers = {'Range': f'bytes={start_byte}-{end_byte}'}

        print("Downloading...")
        response = requests.get(url, headers=headers, stream=True, timeout=120)

        if response.status_code not in [200, 206]:
            print(f"Error: Server returned status {response.status_code}")
            return 0

        # Download the chunk
        compressed_data = b""
        downloaded = 0

        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                compressed_data += chunk
                downloaded += len(chunk)
                if downloaded % (1024 * 1024) == 0:  # Every 1MB
                    print(f"  Downloaded: {downloaded / (1024 * 1024):.1f} MB", end='\r')

        print(f"\n✓ Downloaded {downloaded / (1024 * 1024):.2f} MB (compressed)")

        # Decompress
        print("\nDecompressing...")
        try:
            decompressed = gzip.decompress(compressed_data)
            print(f"✓ Decompressed to {len(decompressed) / (1024 * 1024):.2f} MB")

        except Exception as e:
            print(f"Decompression failed: {e}")
            print("The downloaded chunk may not be a complete gzip block.")
            return 0

        # Extract and clean lines
        print("\nExtracting lines...")
        text = decompressed.decode('utf-8', errors='ignore')
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        print(f"✓ Found {len(lines):,} non-empty lines in chunk")

        # Randomly sample
        if len(lines) <= sample_size:
            print(f"  (Keeping all {len(lines):,} lines)")
            selected_lines = lines
        else:
            print(f"  Randomly sampling {sample_size:,} lines...")
            selected_lines = random.sample(lines, sample_size)

        print(f"✓ Sampled {len(selected_lines):,} lines")

        # Save to file
        SPANISH_SUBTITLES.mkdir(parents=True, exist_ok=True)
        output_file = SPANISH_SUBTITLES / "spanish_opus_random_sample.txt"

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(selected_lines))

        file_size = output_file.stat().st_size
        print(f"\n✓ Saved to: {output_file}")
        print(f"  File size: {file_size / 1024:.1f} KB")
        print(f"  Lines: {len(selected_lines):,}")

        # Show preview
        print(f"\n{'='*70}")
        print("Preview of 15 random samples:")
        print('='*70)
        preview = random.sample(selected_lines, min(15, len(selected_lines)))
        for i, line in enumerate(preview, 1):
            print(f"{i:3d}. {line[:67]}{'...' if len(line) > 67 else ''}")
        print('='*70)

        # Show some statistics
        avg_length = sum(len(line) for line in selected_lines) / len(selected_lines)
        print(f"\nStatistics:")
        print(f"  Average line length: {avg_length:.1f} characters")
        print(f"  Shortest line: {min(len(line) for line in selected_lines)} chars")
        print(f"  Longest line: {max(len(line) for line in selected_lines)} chars")

        return len(selected_lines)

    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Download random Spanish subtitle samples from OPUS'
    )
    parser.add_argument(
        '--chunk-mb',
        type=int,
        default=20,
        help='Chunk size to download in MB (default: 20)'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=10000,
        help='Number of random lines to sample (default: 10000)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    print("="*70)
    print("Random Spanish Subtitle Sampler")
    print("="*70 + "\n")

    lines = download_random_spanish_sample(
        chunk_size_mb=args.chunk_mb,
        sample_size=args.sample_size,
        seed=args.seed
    )

    if lines > 0:
        print(f"\n{'='*70}")
        print("✓ SUCCESS!")
        print(f"{'='*70}")
        print(f"Downloaded {lines:,} random Spanish subtitle lines")
        print("This is authentic, diverse conversational data from movies/TV")
        print("Ready for MWE extraction!")
    else:
        print("\n✗ Download failed")
