#!/usr/bin/env python3
"""
Tifinagh OCR Data Generation Pipeline
Handles dictionary creation, cleaning, synthetic data generation, and format conversion
"""

import os
import re
import json
import random
import string
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Optional
import argparse
from tqdm import tqdm

from vocab_config import filter_tifinagh_text, get_vocabulary, is_tifinagh_char

class TifinaghDataGenerator:
    def __init__(self, corpus_dir: str, output_dir: str, fonts_dir: str):
        self.corpus_dir = Path(corpus_dir)
        self.output_dir = Path(output_dir)
        self.fonts_dir = Path(fonts_dir)
        self.dict_dir = self.output_dir / "dict"
        self.images_dir = self.output_dir / "images"
        
        # Create directories
        self.dict_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
    
    def create_dictionary_from_corpus(self, corpus_files: List[str], output_file: str = "corpus_dict.txt"):
        """Create dictionary from corpus files"""
        print("Creating dictionary from corpus files...")
        
        words = set()
        
        for corpus_file in corpus_files:
            corpus_path = self.corpus_dir / corpus_file
            if not corpus_path.exists():
                print(f"Warning: Corpus file {corpus_path} not found")
                continue
            
            print(f"Processing {corpus_path}...")
            with open(corpus_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # Split into words and add to set
                        line_words = re.findall(r'\S+', line)
                        words.update(line_words)
        
        # Write dictionary
        dict_path = self.dict_dir / output_file
        with open(dict_path, 'w', encoding='utf-8') as f:
            for word in sorted(words):
                f.write(f"{word}\n")
        
        print(f"Dictionary created with {len(words)} words: {dict_path}")
        return dict_path
    
    def clean_dictionary(self, dict_file: str, output_file: str = "clean_dict.txt", 
                        min_length: int = 2, max_length: int = 50):
        """Clean dictionary by filtering valid Tifinagh words"""
        print(f"Cleaning dictionary: {dict_file}...")
        
        dict_path = self.dict_dir / dict_file
        clean_dict_path = self.dict_dir / output_file
        
        if not dict_path.exists():
            raise FileNotFoundError(f"Dictionary file {dict_path} not found")
        
        clean_words = []
        total_words = 0
        
        with open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                total_words += 1
                word = line.strip()
                if word:
                    filtered_word = filter_tifinagh_text(word, min_length, max_length)
                    if filtered_word:
                        clean_words.append(filtered_word)
        
        # Write cleaned dictionary
        with open(clean_dict_path, 'w', encoding='utf-8') as f:
            for word in clean_words:
                f.write(f"{word}\n")
        
        print(f"Dictionary cleaned: {total_words} -> {len(clean_words)} words")
        print(f"Clean dictionary saved: {clean_dict_path}")
        return clean_dict_path
    
    def generate_synthetic_data(self, dict_file: str, num_samples: int = 10000, 
                              split: str = "train", background_type: int = 0):
        """Generate synthetic data using TRDG"""
        print(f"Generating {num_samples} synthetic {split} samples...")
        
        dict_path = self.dict_dir / dict_file
        if not dict_path.exists():
            raise FileNotFoundError(f"Dictionary file {dict_path} not found")
        
        # Create output directory for this split
        split_dir = self.images_dir / split
        split_dir.mkdir(exist_ok=True)
        
        # Get font files
        font_files = list(self.fonts_dir.glob("*.ttf")) + list(self.fonts_dir.glob("*.otf"))
        if not font_files:
            raise FileNotFoundError(f"No font files found in {self.fonts_dir}")
        
        # Prepare TRDG command with absolute paths (using direct trdg executable)
        cmd = [
            "python", "-m", "trdg.run",
            "-c", str(num_samples),
            "-w", "1",  # Number of words per image
            "-t", "2",  # Thread count
            "-f", "50",  # Font size
            "-k", "2",  # Skewing angle
            "-rk",  # Random skewing
            "-bl", "1",  # Blur
            "-rbl",  # Random blur
            "-tc", "#000000,#888888",  # Text color
            "-na", "2",  # Name format
            "-fd", str(self.fonts_dir.resolve()),
            "-dt", str(dict_path.resolve()),  # Dictionary file with absolute path
            "--output_dir", str(split_dir.resolve())
        ]
        
        print(f"Running TRDG command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.output_dir))
            if result.returncode != 0:
                print(f"TRDG error: {result.stderr}")
                raise RuntimeError(f"TRDG failed with return code {result.returncode}")
            print(f"TRDG completed successfully for {split} split")
        except Exception as e:
            print(f"Error running TRDG: {e}")
            raise
        
        return split_dir
    
    def convert_to_doctr_format(self, images_dir: str, split: str = "train"):
        """Convert TRDG output to DocTR format"""
        print(f"Converting {split} data to DocTR format...")
        
        split_path = Path(images_dir) / split
        if not split_path.exists():
            raise FileNotFoundError(f"Split directory {split_path} not found")
        
        # Find labels file
        labels_file = split_path / "labels.txt"
        if not labels_file.exists():
            raise FileNotFoundError(f"Labels file {labels_file} not found")
        
        # Create annotations dictionary with image filename as key and text as value
        annotations = {}
        
        # Read and process labels file
        with open(labels_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    # TRDG format: image_name text
                    parts = line.split(' ', 1)  # Split on first space only
                    if len(parts) == 2:
                        image_name = parts[0]
                        text = parts[1]
                        
                        # Check if image exists
                        image_path = split_path / image_name
                        if image_path.exists():
                            annotations[image_name] = text
        
        # Save annotations in the new format
        doctr_file = split_path / f"{split}_annotations.json"
        with open(doctr_file, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, ensure_ascii=False, indent=2)
        
        print(f"DocTR annotations saved: {doctr_file} ({len(annotations)} samples)")
        return doctr_file
    
    def generate_complete_dataset(self, corpus_files: List[str], 
                                train_samples: int = 8000, 
                                val_samples: int = 1000,
                                test_samples: int = 1000):
        """Generate complete dataset with train/val/test splits"""
        print("Starting complete dataset generation...")
        
        # Step 1: Create dictionary from corpus
        dict_file = self.create_dictionary_from_corpus(corpus_files)
        
        # Step 2: Clean dictionary
        clean_dict_file = self.clean_dictionary(dict_file.name)
        
        # Step 3: Generate synthetic data for each split
        splits = {
            "train": train_samples,
            "val": val_samples,
            "test": test_samples
        }
        
        for split, num_samples in splits.items():
            if num_samples > 0:
                # Generate synthetic images
                split_dir = self.generate_synthetic_data(
                    clean_dict_file.name, 
                    num_samples, 
                    split,
                    background_type=random.randint(0, 3)  # Random background
                )
                
                # Convert to DocTR format
                self.convert_to_doctr_format(self.images_dir, split)
        
        print("Dataset generation completed!")
        print(f"Output directory: {self.output_dir}")
        print(f"Dictionary: {clean_dict_file}")
        print(f"Images: {self.images_dir}")

def main():
    parser = argparse.ArgumentParser(description="Generate Tifinagh OCR training data")
    parser.add_argument("--corpus_dir", type=str, required=True, help="Directory containing corpus files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for generated data")
    parser.add_argument("--fonts_dir", type=str, required=True, help="Directory containing Tifinagh fonts")
    parser.add_argument("--corpus_files", nargs="+", required=True, help="List of corpus files to process")
    parser.add_argument("--train_samples", type=int, default=8000, help="Number of training samples")
    parser.add_argument("--val_samples", type=int, default=1000, help="Number of validation samples")
    parser.add_argument("--test_samples", type=int, default=1000, help="Number of test samples")
    
    args = parser.parse_args()
    
    # Create data generator
    generator = TifinaghDataGenerator(
        corpus_dir=args.corpus_dir,
        output_dir=args.output_dir,
        fonts_dir=args.fonts_dir
    )
    
    # Generate complete dataset
    generator.generate_complete_dataset(
        corpus_files=args.corpus_files,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        test_samples=args.test_samples
    )

if __name__ == "__main__":
    main()