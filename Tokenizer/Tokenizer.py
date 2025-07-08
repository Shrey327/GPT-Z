corpus = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

print("Trainign corpus")
for doc in corpus:
    print(doc)

unique_chars = set()
for doc in corpus:
    for char in doc:
        unique_chars.add(char)

vocab = list(unique_chars)
vocab.sort()

end_of_word = "</w>"
vocab.append(end_of_word)

print("Vocabulary")
print(vocab)
print(f"Vocabulary size: {len(vocab)}")

word_split = {}
for doc in corpus:
    words = doc.split()
    for word in words:
        if word:
            char_list = list(word)+[end_of_word]

            word_tuple = tuple(char_list)
            if word_tuple not in word_split:
                word_split[word_tuple] = 0
            word_split[word_tuple]+=1

print("Word split")
print(word_split)

import collections

def get_pair_stats(splits):
    """Counts the frequency of adjacent pairs in the word splits."""
    pair_counts = collections.defaultdict(int)
    for word_tuple, freq in splits.items():
        symbols = list(word_tuple)
        for i in range(len(symbols)-1):
            pair = (symbols[i], symbols[i+1])
            pair_counts[pair] += freq
    return pair_counts

pair_stats = get_pair_stats(word_split)

print("Pair stats")
print(pair_stats)

def merge_pair(pair_to_merge, splits):
    """Merges a pair of symbols into a single symbol."""
    new_splits = {}
    (first, second) = pair_to_merge
    merged_token = first + second
    for word_tuple, freq in splits.items():
        symbols = list(word_tuple)
        new_symbols = []
        i = 0
        while i < len(symbols):
            if i<len(symbols)-1 and symbols[i] == first and symbols[i+1] == second:
                new_symbols.append(merged_token)
                i+=2
            else:
                new_symbols.append(symbols[i])
                i+=1
        new_splits[tuple(new_symbols)] = freq
    return new_splits
    
num_merges = 15

merges = {}
current_splits = word_split.copy()

print("\n--- Starting BPE Merges ---")
print(f"Initial Splits: {current_splits}")
print("-"*30)

for i in range(num_merges):
    print(f"Iteration {i+1}/{num_merges}")

    pair_stats = get_pair_stats(current_splits)
    if not pair_stats:
        print("No more pairs to merge")
        break
    
    sorted_pairs = sorted(pair_stats.items(), key=lambda x: x[1], reverse=True)
    print(f"Top 5 pairs: {sorted_pairs[:5]}")

    best_pair = max(pair_stats, key=pair_stats.get)
    best_freq = pair_stats[best_pair]
    print(f"Found Best Pair: {best_pair} with frequency {best_freq}")

    current_splits = merge_pair(best_pair, current_splits)
    new_token = best_pair[0] + best_pair[1]
    print(f"Merged {best_pair} into {new_token}")
    print(f"splits after merge: {current_splits}")

    vocab.append(new_token)
    print(f"updated vocabulary: {vocab}")

    merges[best_pair] = new_token
    print(f"updated merges: {merges}")

    print("-"*30)

print("\n --- BPE Merges Complete ---")

print("\n --- Final Vocabulary ---")
print(vocab)
print("\n --- Final Vocabulary size ---")
print({len(vocab)})
for pair, token in merges.items():
    print(f"{pair} -> '{token}'")

print("\nFinal Word Splits after all merges:")
print(current_splits)

print("\nFinal Vocabulary (sorted):")
# Sort for consistent viewing
final_vocab_sorted = sorted(list(set(vocab))) # Use set to remove potential duplicates if any step introduced them
print(final_vocab_sorted)



