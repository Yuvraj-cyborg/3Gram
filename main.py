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
    return("<start>","<start>"+tokens+"<end>")

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


