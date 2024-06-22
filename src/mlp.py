import torch
import typing
import os
import tqdm
import pickle
import optparse

from torch.utils.data import Dataset

ctoi = None
itoc = None

def load_data(input_file: str, block_size: int) -> typing.Tuple:
    global ctoi
    global itoc

    with open(input_file, "r") as file:
        data = file.read()

    words = data.splitlines()

    chars = sorted(list(set("".join(words))))
    chars.insert(0, ".")

    vocab_size = len(chars)

    ctoi = { c: i for i, c in enumerate(chars) }
    itoc = { i: c for c, i in ctoi.items() }

    X = []
    Y = []

    i = 0

    for word in words:
        ctx = [0] * block_size

        for char in word + '.':
            ix = ctoi[char]
            X.append(ctx)
            Y.append(ix)

            ctx = ctx[1:] + [ix]

    return X, Y, vocab_size

class Linear():

    def __init__(self, fan_in: int, fan_out: int, bias: bool = True) -> None:
        self.weight = torch.randn((fan_in, fan_out)) / (fan_in ** 0.5)
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.out = x @ self.weight

        if self.bias is not None:
            self.out += self.bias

        return self.out

    def parameters(self) -> typing.List:
        return [self.weight] + ([] if self.bias is None else [self.bias])

class BatchNorm1d:

    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        self.eps = eps
        self.momentum = momentum
        self.training = True

        # parameters
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

        # buffers (trained with a running momentum update)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            xmean = x.mean(0, keepdim=True) # batch mean
            xvar = x.var(0, keepdim=True, unbiased=True) # batch variance
        else:
            xmean = self.running_mean
            xvar = self.running_var

        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)

        self.out = self.gamma * xhat + self.beta

        # update buffers

        if self.training:
            with torch.no_grad():
                self.running_mean = (1.0 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1.0 - self.momentum) * self.running_var + self.momentum * xvar

        return self.out

    def parameters(self) -> typing.List:
        return [self.gamma, self.beta]

class Tanh():

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.out = torch.tanh(x)

        return self.out

    def parameters(self) -> typing.List:
        return []

def main() -> None:
    option_parser = optparse.OptionParser()

    option_parser.add_option("--train", dest="train", default=False, action="store_true")
    option_parser.add_option("--generate", dest="generate", default=False, action="store_true")
    option_parser.add_option("--model-path", dest="model_path", default=None, type="str")

    options, args = option_parser.parse_args()

    block_size = 4

    # data
    print("Loading training data")
    root_dir = os.path.dirname(os.path.dirname(__file__))
    names_file = f"{root_dir}/data/names.txt"

    X, Y, vocab_size = load_data(names_file, block_size)

    tr_len = int(len(X) * 0.9)

    X_train, Y_train = torch.tensor(X[:tr_len]), torch.tensor(Y[:tr_len])
    X_test, Y_test = torch.tensor(X[tr_len:]), torch.tensor(Y[tr_len:])

    print(f"Training data size: {X_train.shape[0]}")

    if options.train:
        n_embeddings = 10
        n_hiddens = 100

        C = torch.randn((vocab_size, n_embeddings))

        layers = [
            Linear(n_embeddings * block_size, n_hiddens),
            BatchNorm1d(n_hiddens),
            Tanh(),
            Linear(n_hiddens, n_hiddens),
            BatchNorm1d(n_hiddens),
            Tanh(),
            Linear(n_hiddens, n_hiddens),
            BatchNorm1d(n_hiddens),
            Tanh(),
            Linear(n_hiddens, n_hiddens),
            BatchNorm1d(n_hiddens),
            Tanh(),
            Linear(n_hiddens, n_hiddens),
            BatchNorm1d(n_hiddens),
            Tanh(),
            Linear(n_hiddens, vocab_size)
        ]

        with torch.no_grad():
            # last layer: make less confident
            layers[-1].weight *= 0.1

            # all other layers: apply gain
            for layer in layers[:-1]:
                if isinstance(layer, Linear):
                    layer.weight *= (5 / 3) # Optimized value for tanh

        parameters = [C] + [p for layer in layers for p in layer.parameters()]

        print(f"Number of parameters: {sum(p.nelement() for p in parameters)}")

        for p in parameters:
            p.requires_grad = True

        # training
        max_steps = 200_000
        batch_size = 128

        lossi = []

        print("Starting training")

        for i in tqdm.tqdm(range(max_steps), desc="Training MLP model"):
            ix = torch.randint(0, X_train.shape[0], (batch_size,))
            Xb, Yb = X_train[ix], Y_train[ix]

            # Forward pass
            emb = C[Xb]
            x = emb.view(emb.shape[0], -1)

            for layer in layers:
                x = layer(x)

            loss = torch.nn.functional.cross_entropy(x, Yb)

            # Backward pass
            for layer in layers:
                layer.out.retain_grad()

            for p in parameters:
                p.grad = None

            loss.backward()

            lr = 0.1 if i < 100_000 else 0.01

            for p in parameters:
                p.data += -lr * p.grad

            lossi.append(loss.item())

        print("Finished training")
        print(f"Final loss: {lossi[-1]}")

        if options.model_path is not None:
            print(f"Saving model")

            layers.insert(0, C)

            with open(options.model_path, "wb") as file:
                pickle.dump(layers, file)

            layers.pop(0)

            print(f"Saved model to: {options.model_path}")

    if options.generate:
        if options.model_path is None and not options.train:
            print("No model path has been passed to load, exiting")
            return

        if not options.train:
            with open(options.model_path, "rb") as file:
                layers = pickle.load(file)

            C = layers.pop(0)

        for layer in layers:
            if isinstance(layer, BatchNorm1d):
                layer.training = False

        print("Generating from the model")

        for _ in range(20):
            out = []
            ctx = [0] * block_size

            while True:
                emb = C[torch.tensor([ctx])]

                x = emb.view(emb.shape[0], -1)

                for layer in layers:
                    x = layer(x)

                probs = torch.nn.functional.softmax(x, dim=1)

                ix = torch.multinomial(probs, num_samples=1).item()

                ctx = ctx[1:] + [ix]

                out.append(itoc[ix])

                if ix == 0:
                    break

            print("".join(out))

if __name__ == "__main__":
    main()
