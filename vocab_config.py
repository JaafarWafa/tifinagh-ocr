#!/usr/bin/env python3
"""
Tifinagh Vocabulary Configuration
Defines character sets and vocabulary for Tifinagh OCR
"""

# Basic Tifinagh Unicode characters
TIFINAGH_BASIC = [
    'ⴰ', 'ⴱ', 'ⴳ', 'ⴷ', 'ⴹ', 'ⴻ', 'ⴼ', 'ⴽ', 'ⵀ', 'ⵃ', 'ⵄ', 'ⵅ', 'ⵇ', 'ⵉ', 'ⵊ', 'ⵍ',
    'ⵎ', 'ⵏ', 'ⵓ', 'ⵔ', 'ⵕ', 'ⵖ', 'ⵙ', 'ⵚ', 'ⵛ', 'ⵜ', 'ⵟ', 'ⵡ', 'ⵢ', 'ⵣ', 'ⵥ'
]


# Punctuation and special characters
PUNCTUATION = [' ', '.', ',', '!', '?', ':', ';', '-', '(', ')', '[', ']', '{', '}', '"', "'", '/', '\\']

# Numbers
NUMBERS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Latin characters (for mixed text)
LATIN_BASIC = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]

# Complete vocabulary
VOCABULARY = TIFINAGH_BASIC + PUNCTUATION + NUMBERS + LATIN_BASIC

# Character to index mapping
CHAR_TO_IDX = {char: idx for idx, char in enumerate(VOCABULARY)}
IDX_TO_CHAR = {idx: char for idx, char in enumerate(VOCABULARY)}

# Special tokens
BLANK_TOKEN = '<BLANK>'
UNK_TOKEN = '<UNK>'
PAD_TOKEN = '<PAD>'

# Add special tokens to vocabulary
SPECIAL_TOKENS = [BLANK_TOKEN, UNK_TOKEN, PAD_TOKEN]
FULL_VOCABULARY = VOCABULARY + SPECIAL_TOKENS

# Update mappings with special tokens
FULL_CHAR_TO_IDX = {char: idx for idx, char in enumerate(FULL_VOCABULARY)}
FULL_IDX_TO_CHAR = {idx: char for idx, char in enumerate(FULL_VOCABULARY)}

def get_vocabulary():
    """Return the complete vocabulary list"""
    return FULL_VOCABULARY

def get_char_to_idx():
    """Return character to index mapping"""
    return FULL_CHAR_TO_IDX

def get_idx_to_char():
    """Return index to character mapping"""
    return FULL_IDX_TO_CHAR

def get_vocab_size():
    """Return vocabulary size"""
    return len(FULL_VOCABULARY)

def is_tifinagh_char(char):
    """Check if character is Tifinagh"""
    return char in TIFINAGH_BASIC 

def filter_tifinagh_text(text, min_length=2, max_length=50):
    """Filter text to contain only valid Tifinagh characters and meet length requirements"""
    # Remove characters not in vocabulary
    filtered_chars = [char for char in text if char in VOCABULARY]
    filtered_text = ''.join(filtered_chars)
    
    # Check if text contains at least one Tifinagh character
    has_tifinagh = any(is_tifinagh_char(char) for char in filtered_text)
    
    # Check length requirements
    if len(filtered_text) < min_length or len(filtered_text) > max_length:
        return None
    
    if not has_tifinagh:
        return None
    
    return filtered_text

if __name__ == "__main__":
    print(f"Vocabulary size: {get_vocab_size()}")
    print(f"Tifinagh basic characters: {len(TIFINAGH_BASIC)}")
    print(f"Sample vocabulary: {VOCABULARY[:10]}")