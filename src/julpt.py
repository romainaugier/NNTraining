import typing
import sys

from torch.autograd import forward_ad
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import common

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Torch device: {DEVICE}")
# DEVICE = "cpu"

def load_and_prep_data(input_file: str) -> typing.Tuple[typing.List[str], typing.Dict[str, int]]:
    print("Loading data")

    with open(input_file, "r", encoding="utf-8") as file:
        data = file.read()

    tokens = list(data)

    vocab = { k: 0 for k in sorted(list(set(tokens))) }

    return tokens, vocab

def get_vocab_stats(vocab: typing.Dict[str, int], top_tokens_size: int = 50) -> None:
    print(f"Vocab statistics")
    print(f"Vocab size: {len(vocab)}")

    vocab_sorted = sorted(vocab.items(), key=lambda x: x[1], reverse=True)

    print(f"Top {top_tokens_size} tokens")

    for i in range(top_tokens_size):
        print(vocab_sorted[i])

def encode_tokens(tokens: typing.List[str],
                  vocab: typing.Dict[str, int]) -> typing.Tuple[typing.List[int], typing.Dict[str, int], typing.Dict[int, str]]:
    print("Encoding tokens")

    encode_table = dict()
    decode_table = dict()

    for i, token in enumerate(vocab.keys()):
        encode_table[token] = i
        decode_table[i] = token

    encoded_tokens = list()

    for token in tokens:
        encoded_tokens.append(encode_table[token])

    return encoded_tokens, encode_table, decode_table

def get_batch(tokens: torch.Tensor, batch_size: int, ctx_size: int = 10):
    ix = torch.randint(len(tokens) - ctx_size, (batch_size,))

    X = torch.stack([tokens[i : i + ctx_size] for i in ix])
    Y = torch.stack([tokens[i + 1 : i + ctx_size + 1] for i in ix])

    X, Y = X.to(DEVICE), Y.to(DEVICE)

    return X, Y

@torch.no_grad()
def estimate_loss(model: nn.Module,
                  training_tokens: torch.Tensor,
                  val_tokens: torch.Tensor,
                  batch_size: int,
                  ctx_size: int) -> typing.Dict:
    model.eval()

    out = dict()

    losses = torch.zeros(200)

    for k in range(200):
        X, Y = get_batch(training_tokens, batch_size, ctx_size)
        _, loss = model(X, Y)

        losses[k] = loss.item()

    out["train"] = losses.mean()

    for k in range(200):
        X, Y = get_batch(val_tokens, batch_size, ctx_size)
        _, loss = model(X, Y)

        losses[k] = loss.item()

    out["val"] = losses.mean()

    model.train()

    return out

class Head(nn.Module):

    def __init__(self, head_size: int, n_embeddings: int, context_size: int) -> None:
        super().__init__()

        self.key = nn.Linear(n_embeddings, head_size, bias=False)
        self.query = nn.Linear(n_embeddings, head_size, bias=False)
        self.value = nn.Linear(n_embeddings, head_size, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones(context_size, context_size)))

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)

        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads: int, head_size: int, n_embeddings: int, context_size: int) -> None:
        super().__init__()

        self.heads = nn.ModuleList([Head(head_size, n_embeddings, context_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embeddings, n_embeddings)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):

    def __init__(self, n_embeddings: int) -> None:
        super().__init__()

        self.net = nn.Sequential(
                nn.Linear(n_embeddings, 4 * n_embeddings),
                nn.ReLU(),
                nn.Linear(4 * n_embeddings, n_embeddings),
            )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):

    def __init__(self, n_embeddings: int, n_head: int, context_size: int) -> None:
        super().__init__()

        head_size = n_embeddings // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embeddings, context_size)
        self.ffwd = FeedForward(n_embeddings)
        self.ln1 = nn.LayerNorm(n_embeddings)
        self.ln2 = nn.LayerNorm(n_embeddings)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class JulPT(nn.Module):

    def __init__(self, vocab_size: int, context_size: int, n_embeddings: int, n_layers: int) -> None:
        super().__init__()

        self.context_size = context_size

        self.token_embedding_table = nn.Embedding(vocab_size, n_embeddings)
        self.position_embedding_table = nn.Embedding(context_size, n_embeddings)

        self.blocks = nn.Sequential(*[Block(n_embeddings, 4, context_size) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embeddings)

        self.lm_head = nn.Linear(n_embeddings, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE))

        x = token_emb + pos_emb

        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)

            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens: int):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_size:]

            logits, loss = self(idx_cond)

            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx

if __name__ == "__main__":
    tokens, vocab = load_and_prep_data(f"{common.DATA_DIR}/lyrics_jul.txt")
    vocab_size = len(vocab)

    encoded_tokens, encode_table, decode_table = encode_tokens(tokens, vocab)
    encoded_tokens = torch.tensor(encoded_tokens)

    context_size = 128
    batch_size = 64

    jpt = JulPT(vocab_size, context_size, 128, 6)
    jpt = jpt.to(DEVICE)

    training_size = int(len(encoded_tokens) * 0.8)
    val_size = int(len(encoded_tokens) * 0.1)

    training_tokens = encoded_tokens[:training_size]
    val_tokens = encoded_tokens[training_size:training_size + val_size]
    test_tokens = encoded_tokens[training_size + val_size:]

    print("Training the model")

    optimizer = torch.optim.AdamW(jpt.parameters(), lr=1e-4)

    num_training_steps = 5000

    for step in range(num_training_steps):
        if step % int(num_training_steps / 20) == 0:
            losses = estimate_loss(jpt, training_tokens, val_tokens, batch_size, context_size)
            print(f"Step [{step}/{num_training_steps}]: train loss {losses['train']} | val loss {losses['val']}")

        xb, yb = get_batch(training_tokens, batch_size, context_size)

        logits, loss = jpt(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    losses = estimate_loss(jpt, training_tokens, val_tokens, batch_size, context_size)
    print(f"Final loss: train loss {losses['train']} | val loss {losses['val']}")

    print("Saving the model")

    torch.save(jpt.state_dict(), "./models/jpt1.model")

    print("Generating from the model")

    idx = torch.tensor([encode_table['\n']])
    idx = idx.reshape((1, 1))
    idx = idx.to(DEVICE)

    gen = jpt.generate(idx, 1000)

    print("".join([decode_table[t] for t in gen[0].tolist()]))
