import torch

def load_text(filepath:str)->str:
    """Load text data drake_lyrics.txt."""
    with open('/data/drake_lyrics.txt', 'r') as f:
        return f.read()


def create_vocab(text:str):
    """vocab mapping from characters."""
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)} #list of ch to token index
    itos = {i: ch for i, ch in enumerate(chars)} #list of token index to ch
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    return stoi, itos, encode, decode

def encode(text:str, stoi:dict)->list[int]:
    """Encode txt->token ids"""
    return [stoi[c] for c in text]

def decode(tokens:list[int], itos:dict)-> str:
    """."""
    return ''.join([itos[i] for i in tokens])

def get_batch(data: torch.Tensor, block_size:int, batch_size: int, device: str):
    """bath of input, target pairs"""
    ix = torch.randint(len(data)-block_size-1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)


with open('../data/drake_lyrics.txt', 'r') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)
