
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from scr.model import BigramLanguageModel
from scr.data import load_text, create_vocab, encode, decode, split_data, get_batch

#mode parameters (NEEDS TO MATCH TRAINING)
n_embd = 64
n_head = 4
n_layer = 4
block_size = 32
dropout = 0.1

#loading vocab
test = load_text('data/drake_lyrics.txt')
chars, stoi, itos = create_vocab(test)


#recreate model
model = BigramLanguageModel(
    vocab_size=len(chars),
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    block_size=block_size,
    dropout=dropout
)
model.load_state_dict(torch.load('data/drake_gpt_weights.pth',map_location = 'cpu'))
model.eval()

#generate lyrics
context = torch.zeros((1, 1), dtype=torch.long)  # start with a single token 
generated = model.generate(context, max_new_tokens=500, temperature=0.8)
print(decode(generated[0].tolist(), itos))