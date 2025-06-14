#####


import torch
from scr.data import load_text, create_vocab, encode, decode, split_data, get_batch
from scr.model import BigramLanguageModel

#parameters
batch_size = 16
block_size = 32
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'


#load and process
text = load_text('data/drake_lyrics.txt')
chars, stoi, itos = create_vocab(text)
data = torch.tensor(encode(text, stoi), dtype=torch.long)
train_data, val_data = split_data(data, split_ratio=0.9)

def get_batch_wrapper(split):
    d = train_data if split == 'train' else val_data
    return get_batch(d, block_size, batch_size, device)



@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zero(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch_wrapper(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

#MODEL ONE

model = BigramLanguageModel(
    vocab_size=len(chars),
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    block_size=block_size,
    dropout=dropout
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch_wrapper('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
#save weis
torch.save(model.state_dict(), 'drake_gpt_weights.pth')

#sample
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=200, temperature=0.8)[0].tolist(), itos))