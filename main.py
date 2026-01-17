from collections import defaultdict
import random

dataset = [
    "this is a machine",
    "this is a machine",
    "this is a machine",
    "this is a laptop",
    "this is a laptop",
    "a laptop is portable",
    "a machine is powerful",
    "a machine has power",
    "a laptop has battery",
]

def tokenise(sentence):
    """Split sentence into tokens and then add 2 start tokens so that even the first word can be represented as tri-gram and an end token"""
    tokens = sentence.lower().split()
    return["<start>","<start>"+tokens+"<end>"]

def build_vocab(dataset):
    """forming a set by using split words and storing them"""
    vocab = set()
    for sentence in dataset:
        tokens = tokenise(sentence)
        vocab.update(tokens)
    return vocab

def token_id(vocab):
    """each word would have its own id, so its easier to not repeat sequences"""
    token_to_id = {}
    id_to_token = {}

    for idx, token in enumerate(sorted(vocab)):
        token_to_id[token] = idx
        id_to_token[idx] = token

    return token_to_id, id_to_token

def encode(sentence, token_to_id):
    tokens = tokenize(sentence)
    return [token_to_id[token] for token in tokens]

def train(dataset, token_to_id):
    unigram = defaultdict(int)
    bigram = defaultdict(lambda: defaultdict(int))
    trigram = defaultdict(lambda: defaultdict(int))

    for sentence in dataset:
        ids = encode(sentence, token_to_id)

        for i in range(len(ids) - 2):
            w1, w2, w3 = ids[i], ids[i+1], ids[i+2]

            unigram[w3] += 1
            bigram[w2][w3] += 1
            trigram[(w1, w2)][w3] += 1

    return unigram, bigram, trigram

def get_trigram_probs(w1, w2, trigram):
    """calculate trigram probabilities"""
    counts = trigram.get((w1, w2), None)
    if not counts:
        return None

    total = sum(counts.values())
    return {w: c / total for w, c in counts.items()}

def get_unigram_probs(unigram):
    """calculate unigram probabilities"""
    total = sum(unigram.values())
    return {w: c / total for w, c in unigram.items()}

def get_bigram_probs(w2,bigram):
    """calculate bigram probabilities"""
    count = bigram.get(w2, None)
    if not counts:
        return None

    total = sum(counts.values())
    return {w: c / total for w, c in counts.items()}

def predict_next(w1, w2, unigram, bigram, trigram):
    """fallback mechanism if trigram doesnt exist"""
    # Try trigram
    probs = get_trigram_probs(w1, w2, trigram)
    if probs:
        return probs

    # Backoff to bigram
    probs = get_bigram_probs(w2, bigram)
    if probs:
        return probs

    # Backoff to unigram
    return get_unigram_probs(unigram)

