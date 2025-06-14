# data loading- encoding/decoding- vocab-batching

import torch

def load_text(filepath:str)->str:
    """Load text data drake_lyrics.txt"""
    with open('data/drake_lyrics.txt', 'r', encoding='utf-8') as f:
        return f.read()


def create_vocab(text:str):
    """vocab mapping from characters"""
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)} #list of ch to token index
    itos = {i: ch for i, ch in enumerate(chars)} #list of token index to ch
    return chars, stoi, itos

def encode(text: str, stoi: dict)->list[int]:
    """Encode txt->token ids"""
    return [stoi[c] for c in text]

def decode(tokens:list[int], itos:dict)-> str:
    """Decode token ids to txt"""
    return ''.join([itos[i] for i in tokens])


def split_data(data: torch.Tensor, split_ratio: float = 0.9):
    """Split data into train and val sets"""
    n = int(split_ratio*len(data))
    return data[:n], data[n:]


def get_batch(data: torch.Tensor, block_size:int, batch_size: int, device: str):
    """bath of input, target pairs"""
    ix = torch.randint(len(data)-block_size-1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

