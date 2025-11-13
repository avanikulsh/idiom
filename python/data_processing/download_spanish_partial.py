"""
Download PARTIAL Spanish subtitle data from OPUS using HTTP range requests.
Downloads only the first few MB instead of the entire 2GB corpus.
"""
import requests
import gzip
import os
import sys
from pathlib import Path
import io

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from config import SPANISH_SUBTITLES
except ImportError:
    from python.config import SPANISH_SUBTITLES


def download_partial_opus_spanish(chunk_size_mb: int = 5, target_lines: int = 10000):
    """
    Download only the first chunk of OPUS Spanish corpus.

    Args:
        chunk_size_mb: How many MB to download (compressed)
        target_lines: Target number of lines to extract

    Returns:
        Number of lines downloaded
    """
    print(f"Downloading first {chunk_size_mb}MB of Spanish OpenSubtitles corpus...")
    print(f"Target: ~{target_lines:,} lines of real subtitle data")

    url = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/es.txt.gz"

    try:
        # Use HTTP range request to download only first X MB
        chunk_size_bytes = chunk_size_mb * 1024 * 1024
        headers = {'Range': f'bytes=0-{chunk_size_bytes-1}'}

        print(f"\nRequesting first {chunk_size_mb}MB...")
        response = requests.get(url, headers=headers, stream=True, timeout=60)

        if response.status_code not in [200, 206]:  # 206 = Partial Content
            print(f"Error: Server returned status {response.status_code}")
            return 0

        print("✓ Download started")

        # Download the chunk
        compressed_data = b""
        downloaded = 0

        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                compressed_data += chunk
                downloaded += len(chunk)
                if downloaded % (512 * 1024) == 0:  # Every 512KB
                    print(f"  Downloaded: {downloaded / (1024 * 1024):.1f} MB", end='\r')

        print(f"\n✓ Downloaded {downloaded / (1024 * 1024):.2f} MB (compressed)")

        # Decompress the data
        print("\nDecompressing...")
        try:
            decompressed = gzip.decompress(compressed_data)
            print(f"✓ Decompressed to {len(decompressed) / (1024 * 1024):.2f} MB")
        except Exception as e:
            # Partial gzip might not decompress perfectly, try to salvage what we can
            print(f"Partial decompression (expected with chunked download)")
            try:
                # Try decompressing with error tolerance
                decompressor = gzip.GzipFile(fileobj=io.BytesIO(compressed_data))
                decompressed = decompressor.read()
            except:
                print("Could not decompress. Trying alternative approach...")
                return 0

        # Extract lines
        print(f"\nExtracting up to {target_lines:,} lines...")
        text = decompressed.decode('utf-8', errors='ignore')
        lines = text.split('\n')

        print(f"✓ Found {len(lines):,} lines in chunk")

        # Take only what we need
        selected_lines = [line.strip() for line in lines[:target_lines] if line.strip()]

        print(f"✓ Selected {len(selected_lines):,} non-empty lines")

        # Save to file
        SPANISH_SUBTITLES.mkdir(parents=True, exist_ok=True)
        output_file = SPANISH_SUBTITLES / "spanish_opus_sample.txt"

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(selected_lines))

        file_size = output_file.stat().st_size
        print(f"\n✓ Saved to: {output_file}")
        print(f"  File size: {file_size / 1024:.1f} KB")
        print(f"  Lines: {len(selected_lines):,}")

        # Show preview
        print(f"\n{'='*70}")
        print("Preview of first 15 lines:")
        print('='*70)
        for i, line in enumerate(selected_lines[:15], 1):
            print(f"{i:3d}. {line}")
        print('='*70)

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
        description='Download partial Spanish subtitle data from OPUS'
    )
    parser.add_argument(
        '--chunk-mb',
        type=int,
        default=5,
        help='How many MB to download (compressed) (default: 5)'
    )
    parser.add_argument(
        '--target-lines',
        type=int,
        default=10000,
        help='Target number of lines to extract (default: 10000)'
    )

    args = parser.parse_args()

    lines = download_partial_opus_spanish(
        chunk_size_mb=args.chunk_mb,
        target_lines=args.target_lines
    )

    if lines > 0:
        print(f"\n✓ Success! Downloaded {lines:,} real Spanish subtitle lines")
        print("  This is authentic conversational data from movies/TV")
        print("  Ready for MWE extraction!")
    else:
        print("\n✗ Download failed")
