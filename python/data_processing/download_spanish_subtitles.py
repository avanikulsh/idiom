"""
Download a small sample of Spanish subtitle files from OpenSubtitles via OPUS.
Strategic approach: Download just enough to test MWE extraction without overwhelming the laptop.
"""
import requests
import gzip
import os
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from config import SPANISH_SUBTITLES
except ImportError:
    from python.config import SPANISH_SUBTITLES


def download_opus_spanish_sample(max_files: int = 100, max_size_mb: int = 10):
    """
    Download a small sample of Spanish subtitles from OPUS OpenSubtitles corpus.

    Args:
        max_files: Maximum number of subtitle files to download
        max_size_mb: Maximum total size in MB

    Returns:
        Number of files downloaded
    """
    print(f"Downloading up to {max_files} Spanish subtitle files (max {max_size_mb}MB)...")
    print("This will give us enough data to test MWE extraction without overloading your laptop.")

    # OPUS OpenSubtitles Spanish monolingual corpus URL
    # Using the 2018 version which is well-documented
    base_url = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono"

    # Try to download a sample package
    # OPUS provides pre-packaged files by language
    spanish_url = f"{base_url}/es.txt.gz"

    print(f"\nDownloading Spanish subtitle corpus sample from OPUS...")
    print(f"URL: {spanish_url}")

    try:
        response = requests.get(spanish_url, stream=True, timeout=60)
        response.raise_for_status()

        # Check file size
        content_length = int(response.headers.get('content-length', 0))
        size_mb = content_length / (1024 * 1024)

        print(f"Compressed file size: {size_mb:.2f} MB")

        if size_mb > max_size_mb * 2:  # Compressed is usually smaller
            print(f"Warning: File is larger than expected. Proceeding with download...")

        # Download to temp location
        temp_file = "/tmp/spanish_subtitles.txt.gz"

        print("Downloading...")
        with open(temp_file, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded % (1024 * 1024) == 0:  # Every MB
                        print(f"  Downloaded: {downloaded / (1024 * 1024):.1f} MB")

        print(f"\nExtracting and sampling {max_files} files...")

        # Extract and sample lines
        SPANISH_SUBTITLES.mkdir(parents=True, exist_ok=True)

        file_count = 0
        line_buffer = []
        lines_per_file = 100  # Approximate lines per subtitle file
        total_size = 0

        with gzip.open(temp_file, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                line_buffer.append(line)

                # Create a file every N lines
                if len(line_buffer) >= lines_per_file:
                    if file_count >= max_files:
                        break

                    # Save as subtitle file
                    output_file = SPANISH_SUBTITLES / f"spanish_sample_{file_count:04d}.txt"
                    with open(output_file, 'w', encoding='utf-8') as out:
                        out.write('\n'.join(line_buffer))

                    file_size = output_file.stat().st_size
                    total_size += file_size

                    # Check size limit
                    if total_size > max_size_mb * 1024 * 1024:
                        print(f"Reached size limit ({max_size_mb}MB), stopping...")
                        break

                    file_count += 1
                    line_buffer = []

                    if file_count % 10 == 0:
                        print(f"  Created {file_count} files ({total_size / (1024 * 1024):.2f} MB)")

        # Clean up
        os.remove(temp_file)

        print(f"\n✓ Downloaded {file_count} Spanish subtitle samples")
        print(f"  Total size: {total_size / (1024 * 1024):.2f} MB")
        print(f"  Location: {SPANISH_SUBTITLES}")

        return file_count

    except requests.exceptions.RequestException as e:
        print(f"Error downloading from OPUS: {e}")
        print("\nAlternative: You can manually download Spanish subtitles from:")
        print("  - https://www.opensubtitles.org/")
        print("  - https://opus.nlpl.eu/OpenSubtitles-v2018.php")
        return 0
    except Exception as e:
        print(f"Error processing files: {e}")
        return 0


def create_sample_spanish_data():
    """
    Create a small sample Spanish dataset for testing.
    Useful if OPUS download fails.
    """
    print("Creating small sample Spanish dataset for testing...")

    # Some example Spanish sentences with idiomatic expressions
    sample_sentences = [
        "Está lloviendo a cántaros.",  # Raining cats and dogs
        "Me costó un ojo de la cara.",  # Cost an arm and a leg
        "No tiene pelos en la lengua.",  # Doesn't mince words
        "Meter la pata es muy fácil.",  # Put your foot in it
        "Darle la vuelta a la tortilla.",  # Turn the tables
        "Estar en las nubes todo el día.",  # Have your head in the clouds
        "No hay mal que por bien no venga.",  # Every cloud has a silver lining
        "A quien madruga, Dios le ayuda.",  # The early bird catches the worm
        "Más vale tarde que nunca, amigo.",  # Better late than never
        "Tirar la casa por la ventana.",  # Spare no expense
        "Estar entre la espada y la pared.",  # Between a rock and a hard place
        "Tomar el pelo a alguien es malo.",  # Pull someone's leg
        "Dormirse en los laureles no sirve.",  # Rest on your laurels
        "Buscar una aguja en un pajar.",  # Look for a needle in a haystack
        "Pan comido, muy fácil para mí.",  # Piece of cake
    ]

    SPANISH_SUBTITLES.mkdir(parents=True, exist_ok=True)

    # Create a small sample file
    sample_file = SPANISH_SUBTITLES / "spanish_idioms_sample.txt"
    with open(sample_file, 'w', encoding='utf-8') as f:
        for sentence in sample_sentences * 10:  # Repeat for frequency
            f.write(sentence + '\n')

    print(f"✓ Created sample file: {sample_file}")
    print("  This is a tiny test file with Spanish idioms.")
    print("  Run the OPUS downloader for real subtitle data.")

    return 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Download Spanish subtitle sample for MWE extraction'
    )
    parser.add_argument(
        '--max-files',
        type=int,
        default=100,
        help='Maximum number of files to download (default: 100)'
    )
    parser.add_argument(
        '--max-size-mb',
        type=int,
        default=10,
        help='Maximum total size in MB (default: 10)'
    )
    parser.add_argument(
        '--sample-only',
        action='store_true',
        help='Create small test sample instead of downloading'
    )

    args = parser.parse_args()

    if args.sample_only:
        create_sample_spanish_data()
    else:
        files = download_opus_spanish_sample(
            max_files=args.max_files,
            max_size_mb=args.max_size_mb
        )

        if files == 0:
            print("\nDownload failed. Creating sample data instead...")
            create_sample_spanish_data()
