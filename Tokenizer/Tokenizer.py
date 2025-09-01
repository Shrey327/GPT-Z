import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import collections
from typing import List, Dict, Tuple, Optional
import json
import os

class BPETokenizer:
    """Byte Pair Encoding Tokenizer - Based on original Tokenizer.py implementation"""
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.vocab = []
        self.word_to_tokens = {}
        self.token_to_id = {}
        self.id_to_token = {}
        self.merges = {}
        self.end_of_word = "</w>"
        
    def train(self, corpus: List[str]) -> None:
        """Train BPE tokenizer on corpus using original implementation"""
        print("Training BPE tokenizer...")
        
        # Initialize vocabulary with unique characters (from original)
        unique_chars = set()
        for doc in corpus:
            for char in doc:
                unique_chars.add(char)
        
        self.vocab = sorted(list(unique_chars))
        self.vocab.append(self.end_of_word)
        
        # Initialize word splits (from original)
        word_split = {}
        for doc in corpus:
            words = doc.split()
            for word in words:
                if word:
                    char_list = list(word) + [self.end_of_word]
                    word_tuple = tuple(char_list)
                    if word_tuple not in word_split:
                        word_split[word_tuple] = 0
                    word_split[word_tuple] += 1
        
        # Perform BPE merges using original functions
        current_splits = word_split.copy()
        num_merges = min(self.vocab_size - len(self.vocab), 100)
        
        for i in range(num_merges):
            pair_stats = self._get_pair_stats(current_splits)
            if not pair_stats:
                break
                
            best_pair = max(pair_stats, key=pair_stats.get)
            current_splits = self._merge_pair(best_pair, current_splits)
            
            new_token = best_pair[0] + best_pair[1]
            self.vocab.append(new_token)
            self.merges[best_pair] = new_token
        
        # Create token mappings
        self.vocab = sorted(list(set(self.vocab)))
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}
        
        print(f"BPE training complete. Vocabulary size: {len(self.vocab)}")
    
    def _get_pair_stats(self, splits: Dict) -> Dict:
        """Count frequency of adjacent pairs - from original Tokenizer.py"""
        pair_counts = collections.defaultdict(int)
        for word_tuple, freq in splits.items():
            symbols = list(word_tuple)
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pair_counts[pair] += freq
        return pair_counts
    
    def _merge_pair(self, pair_to_merge: Tuple, splits: Dict) -> Dict:
        """Merge a pair of symbols - from original Tokenizer.py"""
        new_splits = {}
        first, second = pair_to_merge
        merged_token = first + second
        
        for word_tuple, freq in splits.items():
            symbols = list(word_tuple)
            new_symbols = []
            i = 0
            while i < len(symbols):
                if (i < len(symbols) - 1 and 
                    symbols[i] == first and symbols[i + 1] == second):
                    new_symbols.append(merged_token)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            new_splits[tuple(new_symbols)] = freq
        return new_splits
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        words = text.split()
        tokens = []
        
        for word in words:
            if word in self.word_to_tokens:
                tokens.extend(self.word_to_tokens[word])
            else:
                # Apply BPE merges to word
                word_tokens = list(word) + [self.end_of_word]
                word_tokens = self._apply_merges(word_tokens)
                self.word_to_tokens[word] = word_tokens
                tokens.extend(word_tokens)
        
        return [self.token_to_id.get(token, 0) for token in tokens]
    
    def _apply_merges(self, tokens: List[str]) -> List[str]:
        """Apply learned BPE merges to tokens"""
        while True:
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            if not pairs:
                break
                
            # Find the pair that appears first in our merge order
            best_pair = None
            for pair in pairs:
                if pair in self.merges:
                    best_pair = pair
                    break
            
            if best_pair is None:
                break
                
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (i < len(tokens) - 1 and 
                    tokens[i] == best_pair[0] and tokens[i + 1] == best_pair[1]):
                    new_tokens.append(self.merges[best_pair])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text"""
        tokens = [self.id_to_token.get(token_id, '<UNK>') for token_id in token_ids]
        text = ''.join(tokens).replace(self.end_of_word, ' ')
        return text.strip()
    
    def save(self, filepath: str) -> None:
        """Save tokenizer to file"""
        data = {
            'vocab': self.vocab,
            'merges': {str(k): v for k, v in self.merges.items()},
            'token_to_id': self.token_to_id,
            'id_to_token': self.id_to_token,
            'vocab_size': self.vocab_size
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str) -> None:
        """Load tokenizer from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.vocab = data['vocab']
        self.merges = {eval(k): v for k, v in data['merges'].items()}
        self.token_to_id = data['token_to_id']
        self.id_to_token = {int(k): v for k, v in data['id_to_token'].items()}
        self.vocab_size = data['vocab_size']


# Example usage and testing
if __name__ == "__main__":
    # Test the BPE tokenizer
    corpus = [
        "This is the first document.",
        "This document is the second document.",
        "And this is the third one.",
        "Is this the first document?",
    ]
    
    print("Testing BPE Tokenizer")
    print("=" * 30)
    
    tokenizer = BPETokenizer(vocab_size=50)
    tokenizer.train(corpus)
    
    # Test encoding and decoding
    test_text = "This is a test document."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"Original: '{test_text}'")
    print(f"Encoded: {encoded}")
    print(f"Decoded: '{decoded}'")
    print(f"Vocabulary size: {len(tokenizer.vocab)}")
