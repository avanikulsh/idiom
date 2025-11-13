"""
Utility functions for parsing subtitle files (SRT, VTT formats).
"""
import pysrt
from pathlib import Path
from typing import List, Dict
import re


def parse_srt(file_path: str) -> List[str]:
    """
    Parse an SRT subtitle file and extract text.

    Args:
        file_path: Path to the SRT file

    Returns:
        List of subtitle text lines
    """
    try:
        subs = pysrt.open(file_path, encoding='utf-8')
        return [sub.text.replace('\n', ' ') for sub in subs]
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return []


def clean_subtitle_text(text: str) -> str:
    """
    Clean subtitle text by removing formatting tags and extra whitespace.

    Args:
        text: Raw subtitle text

    Returns:
        Cleaned text
    """
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove speaker labels like [SPEAKER]:
    text = re.sub(r'\[.*?\]:', '', text)
    text = re.sub(r'\(.*?\)', '', text)

    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def load_subtitles_from_directory(directory: Path, file_extension: str = "srt") -> Dict[str, List[str]]:
    """
    Load all subtitle files from a directory.

    Args:
        directory: Path to directory containing subtitle files
        file_extension: File extension to look for (default: "srt")

    Returns:
        Dictionary mapping filename to list of subtitle texts
    """
    subtitles_data = {}

    for file_path in directory.glob(f"*.{file_extension}"):
        texts = parse_srt(str(file_path))
        cleaned_texts = [clean_subtitle_text(text) for text in texts if text.strip()]
        subtitles_data[file_path.name] = cleaned_texts

    print(f"Loaded {len(subtitles_data)} subtitle files from {directory}")
    return subtitles_data


def combine_subtitles(subtitle_list: List[str], window_size: int = 3) -> List[str]:
    """
    Combine adjacent subtitles using a sliding window to capture multi-subtitle MWEs.

    Args:
        subtitle_list: List of individual subtitle texts
        window_size: Number of consecutive subtitles to combine

    Returns:
        List of combined subtitle texts
    """
    combined = []

    for i in range(len(subtitle_list)):
        for w in range(1, min(window_size + 1, len(subtitle_list) - i + 1)):
            combined_text = ' '.join(subtitle_list[i:i+w])
            combined.append(combined_text)

    return combined
