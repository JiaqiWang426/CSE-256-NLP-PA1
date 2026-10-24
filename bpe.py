from collections import Counter

import torch
from torch.nn.utils.rnn import pad_sequence

from sentiment_data import read_sentiment_examples
from utils import Indexer


def _merge_pair(symbols, pair):
    """Merge all non-overlapping occurrences of one pair, from left to right."""
    merged = []
    i = 0
    while i < len(symbols):
        if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
            merged.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            merged.append(symbols[i])
            i += 1
    return tuple(merged)


def train_bpe(infile: str, vocab_size, window_size=2):
    """Learn BPE rules from train text; return (ordered merges, token Indexer).

    vocab_size includes PAD, UNK, the initial alphabet, </w>, and every new
    merged symbol. Earlier symbols remain available for encoding new words.
    Training stops early if there are no adjacent pairs left to merge.
    """
    if window_size != 2:
        raise ValueError("BPE merges adjacent pairs, so window_size must be 2")

    # 1. Count words across the entire training set. Labels are not used.
    exs = read_sentiment_examples(infile)
    word_counts = Counter(word for ex in exs for word in ex.words)

    # 2. Represent each word as characters followed by one end-of-word symbol.
    # Example: {('l', 'o', 'w', '</w>'): 3, ...}
    word_symbols = Counter()
    for word, count in word_counts.items():
        symbols = tuple(list(word) + ["</w>"])
        word_symbols[symbols] += count

    word_indexer = Indexer()
    word_indexer.add_and_get_index("PAD")
    word_indexer.add_and_get_index("UNK")
    alphabet = {"</w>"}
    for symbols in word_symbols:
        alphabet.update(symbols)
    for symbol in sorted(alphabet):
        word_indexer.add_and_get_index(symbol)
    if vocab_size < len(word_indexer):
        raise ValueError(
            f"vocab_size must be at least {len(word_indexer)} "
            "to include the alphabet and special tokens"
        )

    merges = []
    while len(word_indexer) < vocab_size:
        # 3. Count adjacent pairs in the CURRENT representation of every word.
        pair_counts = Counter()
        for symbols, count in word_symbols.items():
            for pair in zip(symbols, symbols[1:]):
                pair_counts[pair] += count
        if not pair_counts:
            break

        # Resolve frequency ties lexicographically so repeated runs agree.
        best_pair = min(pair_counts, key=lambda pair: (-pair_counts[pair], pair))
        merges.append(best_pair)
        word_indexer.add_and_get_index(best_pair[0] + best_pair[1])

        # 4. Apply this one rule to ALL words, then recount on the next iteration.
        updated_words = Counter()
        for symbols, count in word_symbols.items():
            updated_words[_merge_pair(symbols, best_pair)] += count
        word_symbols = updated_words

    return merges, word_indexer


def encode_word(word, merges):
    """Apply the learned rules in training order, without learning new rules."""
    symbols = tuple(list(word) + ["</w>"])
    for pair in merges:
        if len(symbols) < 2:
            break
        symbols = _merge_pair(symbols, pair)
    return list(symbols)


def encode_sentence(words, merges, word_indexer, cache=None):
    """Encode preprocessed words as one sequence of subword IDs."""
    if cache is None:
        cache = {}
    unk_id = word_indexer.index_of("UNK")
    ids = []
    for word in words:
        # Repeated words share their encoding; this never changes the vocabulary.
        if word not in cache:
            word_ids = []
            for subword in encode_word(word, merges):
                index = word_indexer.index_of(subword)
                word_ids.append(index if index != -1 else unk_id)
            cache[word] = word_ids
        ids.extend(cache[word])
    return ids


def bpe_read_X_y(infile, merges, word_indexer, cache=None):
    """Encode and pad a labeled dataset using a fixed, train-fitted tokenizer."""
    examples = read_sentiment_examples(infile)
    if not examples:
        raise ValueError(f"No sentiment examples in {infile}")
    if cache is None:
        cache = {}
    X, y = [], []
    for example in examples:
        ids = encode_sentence(example.words, merges, word_indexer, cache)
        X.append(torch.tensor(ids or [0], dtype=torch.long))
        y.append(example.label)
    return pad_sequence(X, batch_first=True, padding_value=0), torch.tensor(y, dtype=torch.long)

if __name__ == "__main__":
    merges, word_indexer = train_bpe("data/train.txt", vocab_size=500)
    print(encode_word("movie", merges))
    print(encode_sentence(["a", "good", "movie"], merges, word_indexer))
