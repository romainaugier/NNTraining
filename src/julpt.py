import typing
import sys

import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import common

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"

def load_and_prep_data(input_file: str) -> typing.Tuple[typing.List[str], typing.Dict[str, int]]:
    print("Loading data")

    with open(input_file, "r") as file:
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

class JulPT(nn.Module):

    def __init__(self, vocab_size: int) -> None:
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)

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
            logits, loss = self(idx)

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

    context_size = 10
    batch_size = 32

    jpt = JulPT(vocab_size)
    jpt = jpt.to(DEVICE)

    training_size = int(len(encoded_tokens) * 0.8)
    val_size = int(len(encoded_tokens) * 0.1)

    training_tokens = encoded_tokens[:training_size]
    val_tokens = encoded_tokens[training_size:training_size + val_size]
    test_tokens = encoded_tokens[training_size + val_size:]

    print("Training the model")

    optimizer = torch.optim.AdamW(jpt.parameters(), lr=1e-3)

    num_training_steps = 10000

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

    print("Generating from the model")

    idx = torch.tensor([encode_table['\n']])
    idx = idx.reshape((1, 1))
    idx = idx.to(DEVICE)

    gen = jpt.generate(idx, 1000)

    print("".join([decode_table[t] for t in gen[0].tolist()]))
