import typing
import time
import math
import sys
import optparse
import json
import os
import re
import tiktoken
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F

import common

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

re_tokenizer = re.compile(r"[A-Z] |\S+'|\d+|[A-Z\.]{2,20}| [a-zà-ÿ]{2,}|[a-zà-ÿ]{2,}|[a-zà-ÿ]{1}|[A-ZÀ-Ñ][a-zà-ÿ]+|[A-ZÀ-Ñ]|(?:\r\n|\r|\n)")

class Tokenizer():

    def __init__(self, path: str = None) -> None:
        self._table = dict()

        if path is not None and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as file:
                self._table = json.load(file)

    def train(self, text: str, max_tokens: int = 256) -> None:
        raise NotImplementedError

    def encode(self, text: str) -> typing.List[int]:
        raise NotImplementedError

    def decode(self, tokens: typing.List[int]) -> str:
        raise NotImplementedError

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(self._table, file)

class BPE(Tokenizer):

    def train(self, text: str, max_tokens: int = 256) -> typing.List[int]:
        self._table.clear()

        while len(self._table) <= max_tokens:
            occurences = dict()

            for i in range(len(text) - 1):
                pair = text[i : i + 1]

                count = occurences.get(pair, 0)
                occurences[pair] = count + 1

            most_occuring_pair = sorted(list(occurences.items()), key=lambda x: x[1], reverse=True)[0]

            self._table[str(len(self._table) + 1)] = most_occuring_pair

    def encode(self, text: str) -> typing.List[int]:
        pass

    def decode(self, tokens: typing.List[int]) -> str:
        pass

def tokenize(text: str) -> typing.Tuple[typing.List[int], typing.Dict[str, int]]:
    vocab = dict()

    tokens = list()

    for token in re_tokenizer.findall(text):
        tokens.append(token)

        count = vocab.get(token, 0)
        vocab[token] = count + 1

    return tokens, vocab

def load_and_prep_data(input_file: str) -> typing.Tuple[typing.List[str], typing.Dict[str, int]]:
    print("Loading data")

    with open(input_file, "r", encoding="utf-8") as file:
        data = file.read()

    return tokenize(data)

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

    num_estimates = 50

    losses = torch.zeros(num_estimates)

    for k in range(num_estimates):
        X, Y = get_batch(training_tokens, batch_size, ctx_size)
        _, loss = model(X, Y)

        losses[k] = loss.item()

    out["train"] = losses.mean()

    for k in range(num_estimates):
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
        _, T, C = x.shape

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
                nn.Dropout(0.2),
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

    def generate(self, idx, max_new_tokens: int, end_token: int = None):
        idx_next = None
        i = 0

        while (i < max_new_tokens) or (idx_next != end_token):
            idx_cond = idx[:, -self.context_size:]

            logits, loss = self(idx_cond)

            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

            i += 1

            yield idx_next

def main() -> None:
    option_parser = optparse.OptionParser()

    option_parser.add_option("--train", dest="train", action="store_true")
    option_parser.add_option("--generate", dest="generate", action="store_true")
    option_parser.add_option("--model-path", dest="model_path", type="str", default=None)
    option_parser.add_option("--num-tokens", dest="num_tokens", type="int", default=1000)
    option_parser.add_option("--num-training-steps", dest="num_training_steps", type="int", default=10000)
    option_parser.add_option("--tokenizer", dest="tokenizer", type="str", default="r50k_base")

    options, args = option_parser.parse_args()

    encoder = tiktoken.get_encoding(options.tokenizer)

    if options.train:
        if options.model_path is None:
            print("No path has been specified to load the model")
            return

        with open(f"{common.DATA_DIR}/lyrics_jul_2.txt", "r", encoding="utf-8") as file:
            text = file.read()

        encoded_tokens = encoder.encode(text)

        vocab_size = encoder.max_token_value + 1

        encoded_tokens = torch.tensor(encoded_tokens)
        batch_size = 32

        params = {
            "n_embeddings": 384,
            "n_layers": 64,
            "context_size": 256,
            "vocab_size": vocab_size
        }

        print(f"Model parameters for training:")
        print(json.dumps(params, indent=2))

        jpt = JulPT(params["vocab_size"], params["context_size"], params["n_embeddings"], params["n_layers"]).to(DEVICE)

        training_size = int(len(encoded_tokens) * 0.8)
        val_size = int(len(encoded_tokens) * 0.1)

        training_tokens = encoded_tokens[:training_size]
        val_tokens = encoded_tokens[training_size:training_size + val_size]
        test_tokens = encoded_tokens[training_size + val_size:]

        print("Training the model")
        print(f"Number of training steps: {options.num_training_steps}")
        print(f"Expecting initial loss of: {-torch.log(torch.Tensor([[1.0 / float(vocab_size)]])).item()}")

        optimizer = torch.optim.AdamW(jpt.parameters(), lr=1e-4)

        num_training_steps = options.num_training_steps

        num_zfill = math.ceil(math.log10(num_training_steps))

        for step in range(num_training_steps):
            if (step % int(num_training_steps / 10) == 0) and (step > 0):
                losses = estimate_loss(jpt, training_tokens, val_tokens, batch_size, params["context_size"])
                print(f"\n[{time.strftime('%H:%M:%S')}] Step [{str(step).zfill(num_zfill)}/{num_training_steps}]: train loss {losses['train']} | val loss {losses['val']}")
            else:
                print(f"\r[{time.strftime('%H:%M:%S')}] Step [{str(step).zfill(num_zfill)}/{num_training_steps}]", end='')

            xb, yb = get_batch(training_tokens, batch_size, params["context_size"])

            _, loss = jpt(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        losses = estimate_loss(jpt, training_tokens, val_tokens, batch_size, params["context_size"])
        print(f"Final loss: train loss {losses['train']} | val loss {losses['val']}")

        print("Saving the model")

        torch.save(jpt.state_dict(), options.model_path)

        model_dir = os.path.dirname(options.model_path)
        model_name, _ = os.path.splitext(os.path.basename(options.model_path))

        params_file_path = f"{model_dir}/{model_name}_parms.json"

        with open(params_file_path, "w") as file:
            json.dump(params, file)

    if options.generate:
        print("Loading model")

        model_dir = os.path.dirname(options.model_path)
        model_name, _ = os.path.splitext(os.path.basename(options.model_path))
        params_file_path = f"{model_dir}/{model_name}_parms.json"

        with open(params_file_path, "r") as file:
            params = json.load(file)

        print("Generating from the model")

        jpt = JulPT(params["vocab_size"],
                    params["context_size"],
                    params["n_embeddings"],
                    params["n_layers"])

        jpt.load_state_dict(torch.load(options.model_path, map_location=DEVICE))
        jpt.to(DEVICE)

        idx = torch.tensor([encoder.encode('\n')], device=DEVICE)
        idx = idx.reshape((1, 1))
        idx = idx.to(DEVICE)

        for gen in jpt.generate(idx, options.num_tokens, end_token=encoder.encode('\n')[0]):
            sys.stdout.write(encoder.decode([gen]))
            sys.stdout.flush()

if __name__ == "__main__":
    main()
