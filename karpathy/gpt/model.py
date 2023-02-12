import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter


# -----------------------------------------------------------------------------
# Transformer Language Model (*exactly* as used in GPT-2)


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT
    repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU)
    paper: https://arxiv.org/abs/1606.08415
    """

    # noinspection PyMethodMayBeStatic
    def forward(self, x):
        t = math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
        return 0.5 * x * (1.0 + torch.tanh(t))


class CasualSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, n_heads, emb_dim, block_size):
        super().__init__()
        assert emb_dim % n_heads == 0

        # For all heads combined:
        self.key = nn.Linear(emb_dim, emb_dim, bias=False)
        self.query = nn.Linear(emb_dim, emb_dim, bias=False)
        self.value = nn.Linear(emb_dim, emb_dim, bias=False)
        self.n_heads = n_heads

        # causal mask to ensure that attention is only applied to the left in the
        # input sequence
        self.register_buffer(
            "mask",
            torch.ones((block_size, block_size))
            .tril()
            .view(1, 1, block_size, block_size),
        )

        self.c_proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        B, T, C = x.shape

        # Calculate query, key, values for all heads in batch:
        k = self.key(x)  # B, T, C
        q = self.query(x)  # B, T, C
        v = self.value(x)  # B, T, C

        head_size = C // self.n_heads

        # Break up by head: make the shape (B, T, n_heads, head_size)
        k = k.view(B, T, self.n_heads, head_size)
        q = q.view(B, T, self.n_heads, head_size)
        v = v.view(B, T, self.n_heads, head_size)

        # Move head forward to be the batch dim (B, n_heads, T, head_size)
        k = k.transpose(1, 2)  # B, n_heads, T, head_size
        q = q.transpose(1, 2)  # B, nh, T, hs
        v = v.transpose(1, 2)  # B, nh, T, hs

        # Perform causal self-attention
        # (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, hs, hs)
        att = (k @ q.transpose(-2, -1)) * head_size**-0.5

        mask = self.mask[:, :, :T, :T]  # Making sure mask can work with T
        # smaller than the block_size (e.g. during the generation from a single
        # character prompt where T=1)
        att = att.masked_fill(mask == 0, float("-inf"))
        att = att.softmax(-1)

        # (B, nh, hs, hs) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = att @ v

        # Re-assemble all head outputs side by side:
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection:
        y = self.c_proj(y)
        return y


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_heads: int, emb_dim: int, block_size: int):
        super().__init__()
        # Communication part (share info with other tokens):
        self.ln1 = nn.LayerNorm(emb_dim)
        self.attn = CasualSelfAttention(
            n_heads=n_heads,
            emb_dim=emb_dim,
            block_size=block_size,
        )

        # Computation part (fully-connected nn on self):
        self.ln2 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            NewGELU(),
            nn.Linear(emb_dim * 4, emb_dim),  # projection
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nn.Module):
    """Transformer Language Model, exactly as seen in GPT-2"""

    def __init__(
        self,
        vocab_size,
        block_size,
        emb_dim,
        n_layers,
        n_heads,
    ):
        super().__init__()
        self.block_size = block_size
        self.transformer = nn.ModuleDict(
            dict(
                tok_emb=nn.Embedding(vocab_size, emb_dim),
                pos_emb=nn.Embedding(vocab_size, emb_dim),
                blocks=nn.ModuleList(
                    [Block(n_heads, emb_dim, block_size) for _ in range(n_layers)]
                ),
                lnorm=nn.LayerNorm(emb_dim),
            )
        )
        self.lm_head = nn.Linear(emb_dim, vocab_size, bias=False)

        # Report number of parameters (note we don't count the decoder parameters
        # in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print(f"transformer parameters: {n_params}")

    def forward(self, x, targets=None) -> tuple[torch.Tensor, torch.Tensor | None]:
        device = x.device
        b, t = x.shape
        assert t <= self.block_size, (
            f"Cannot forward sequence of length {t}, "
            f"block size is only {self.block_size}"
        )
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # (1, t)

        # forward the GTP model
        tok_emb = self.transformer.tok_emb(x)  # (b, t, c)
        pos_emb = self.transformer.pos_emb(pos)  # (1, t, c)
        x = tok_emb + pos_emb  # (b, t, c)
        for block in self.transformer.blocks:
            x = block(x)
        x = self.transformer.lnorm(x)

        loss = None
        logits = self.lm_head(x)
        if targets is not None:
            b, t, vocab_size = logits.shape
            loss = F.cross_entropy(
                logits.view(b * t, vocab_size),  # (b*t, vocab_size)
                targets.view(b * t),  # (b*t)
                ignore_index=-1,
            )
        return logits, loss


class CharDataset(Dataset):
    def __init__(self, words: list[str]):
        words = self._clean_words(words)
        self.words = words

        chars = sorted(list(set("".join(words))))
        self.vocab_size = len(chars) + 1  # characters followed by special 0 token

        max_word_length: int = max(len(w) for w in words)
        self.block_size = 1 + max_word_length  # <START> token followed by characters

        self.stoi = {s: i + 1 for i, s in enumerate(chars)}
        self.itos = {i: s for s, i in self.stoi.items()}

        print(f"Number of examples in the dataset: {len(words)}")
        print(f"Max word length: {max_word_length}")
        print(f"Number of unique characters in the vocabulary: {len(chars)}")
        print("Vocabulary:", "".join(chars))

        # Partition the input data into a training and the test set
        # 10% of the training set, or up to 1000 examples
        test_set_size: int = min(1000, int(len(words) * 0.1))
        self.train_set, self.test_set = torch.utils.data.random_split(
            self, [len(words) - test_set_size, test_set_size]
        )
        print(
            f"Split the dataset into {len(self.train_set)} training examples "
            f"and {len(self.test_set)} test examples"
        )

    @staticmethod
    def _clean_words(words: list[str]) -> list[str]:
        words = [w.strip() for w in words]  # Get rid of any leading/trailing space
        words = [w for w in words if w]  # Get rid of any empty strings
        return words

    def encode(self, word) -> torch.Tensor:
        return torch.tensor([self.stoi[w] for w in word], dtype=torch.long)

    def decode(self, ints: list[int]) -> str:
        return "".join(self.itos[i] for i in ints)

    def __len__(self):
        return len(self.words)

    def __getitem__(self, index: int):
        word: str = self.words[index]
        w: torch.Tensor = self.encode(word)

        x = torch.zeros(self.block_size, dtype=torch.long)
        y = torch.zeros(self.block_size, dtype=torch.long)
        x[1 : 1 + len(w)] = w
        y[: len(w)] = w
        y[len(w) + 1 :] = -1  # index -1 will mask the loss at the inactive locations
        return x, y


@torch.no_grad()
def generate(
    model: nn.Module,
    x: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    do_sample: bool = False,
    top_k: int | None = None,
):
    """
    Take a conditioning sequence of indices x (LongTensor of shape (b, t))
    and complete the sequence max_new_tokens times, feeding the predictions back
    into the model each time. Most likely you'll want to make sure to be in
    model.eval() mode of operation for this.
    """
    block_size = model.block_size
    for _ in range(max_new_tokens):
        # If the sequence context is growing too long we must crop it at block_size
        x = x if x.shape[1] <= block_size else x[:, -block_size:]

        # Forward the model to get the logits for the index in the sequence
        logits, _ = model(x)

        # Pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature

        # Optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("Inf")

        # Apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)

        # Either sample from the distribution or take the most likely element
        if do_sample:
            x_next = torch.multinomial(probs, num_samples=1)
        else:
            _, x_next = torch.topk(probs, k=1, dim=-1)

        # append sampled index to the running sequence and continue
        x = torch.cat((x, x_next), dim=1)

    return x


def print_samples(
    model: nn.Module,
    dataset: CharDataset,
    device: str,
    num: int = 10,
    top_k: int | None = None,
):
    """
    Samples from the model and pretty prints the decoded samples
    """
    x_init = torch.zeros(num, 1, dtype=torch.long).to(device)

    top_k = top_k if top_k != -1 else None

    # -1 because we already start with <START> token (index 0)
    n_steps = dataset.block_size - 1

    x_sampled = generate(
        model, x_init, max_new_tokens=n_steps, top_k=top_k, do_sample=True
    ).to("cpu")

    train_samples, test_samples, new_samples = [], [], []

    for sample_i in range(x_sampled.shape[0]):
        # Get the sample_i'th row of sampled integers, as a Python list.
        # We are also trimming out the first <START> token.
        row: list[int] = x_sampled[sample_i, 1:].tolist()

        # Token "0" is the <STOP> token, so we crop the output sequence at that point.
        crop_from = row.index(0) if 0 in row else len(row)
        row = row[:crop_from]

        word_sample = dataset.decode(row)

        # separately track samples that we have and have not seen before
        if word_sample in dataset.train_set:
            train_samples.append(word_sample)
        elif word_sample in dataset.test_set:
            test_samples.append(word_sample)
        else:
            new_samples.append(word_sample)

    for samples, desc in [
        (train_samples, "in train"),
        (test_samples, "in test"),
        (new_samples, "new"),
    ]:
        print(f"{len(samples)} samples that are {desc}:", " ".join(samples))


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    dataset: Dataset,
    device: str,
    batch_size: int = 50,
    max_batches: int | None = None,
):
    model.eval()
    losses = []
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    for batch_i, batch in enumerate(loader):
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        _, loss = model(x, y)
        losses.append(loss)
        if max_batches is not None and batch_i >= max_batches:
            break

    out = torch.tensor(losses).mean().item()
    model.train()
    return out


def main():
    n_layers: int = 4
    emb_dim: int = 64
    n_heads: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    resume = False
    sample_only = False
    batch_size = 32
    learning_rate = 5e-4
    weight_decay = 0.01
    num_workers = 1
    max_steps = 100_000

    seed = 0
    input_text_path = Path("../names.txt")

    ############################

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    writer = SummaryWriter()

    # Init dataset
    with input_text_path.open("r") as f:
        words: list[str] = f.read().splitlines()
    dataset = CharDataset(words)

    print(f"Dataset determined that: {dataset.vocab_size=}, {dataset.block_size=}")

    # init model
    model = Transformer(
        vocab_size=dataset.vocab_size,
        block_size=dataset.block_size,
        emb_dim=emb_dim,
        n_heads=n_heads,
        n_layers=n_layers,
    ).to(device)
    print(f"model #params: {sum(p.numel() for p in model.parameters())}")

    model_path = "model.pt"
    if resume or sample_only:
        # note: if we sample-only then we also assume we are resuming
        print("resuming from existing model in the workdir")
        model.load_state_dict(torch.load(model_path))

    if sample_only:
        print_samples(model=model, dataset=dataset, num=50, device=device)
        sys.exit()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    data_loader = DataLoader(
        dataset.train_set,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        sampler=torch.utils.data.RandomSampler(
            dataset.train_set, replacement=True, num_samples=int(1e10)
        ),
    )

    def _evaluate_and_save(best_loss: int | None = None) -> int:
        print_samples(model, dataset, device, num=10)
        train_loss, test_loss = [
            evaluate(model, subset, device, batch_size=100, max_batches=10)
            for subset in [dataset.train_set, dataset.test_set]
        ]
        writer.add_scalar("Loss/train", train_loss, step)
        writer.add_scalar("Loss/test", test_loss, step)
        writer.flush()
        print(f"Step {step} train loss: {train_loss} test loss: {test_loss}")
        # Save the model to disk if it has improved
        if best_loss is None or test_loss < best_loss:
            print(
                f"test loss {test_loss} is the best so far, saving model to "
                f"{model_path}"
            )
            torch.save(model.state_dict(), model_path)
            best_loss = test_loss
        return best_loss

    # Training loop
    best_loss = None
    step = 0
    try:
        for (x, y) in data_loader:
            t0 = time.time()

            x.to(device)
            y.to(device)

            logits, loss = model(x, y)

            # Calculate the gradient, update the weights
            model.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # Wait for all CUDA work on the GPU to finish then calculate iteration time
            # taken
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t1 = time.time()

            # Logging
            if step % 10 == 0:
                print(
                    f"step {step} | loss {loss.item():.4f} | step time "
                    f"{(t1 - t0) * 1000:.2f}ms"
                )

            # Evaluate the model
            if step > 0 and step % 500 == 0:
                best_loss = _evaluate_and_save(best_loss)

            step += 1
            # Termination conditions
            if step >= max_steps:
                break
    except KeyboardInterrupt:
        print("Training interrupted by user")

    print("-" * 100)
    print("Final result:")
    _evaluate_and_save(best_loss)


if __name__ == "__main__":
    main()
