import math
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import random_split, Dataset
from torch.utils.data import DataLoader

from tokenizers.implementations import ByteLevelBPETokenizer
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerState,
    TrainerCallback,
)

datasets_path = (
    next(
        Path(p)
        for p in ["/Users/vlad/googledrive", "/content/drive/MyDrive"]
        if Path(p).exists()
    )
    / "AI/datasets"
)

saves_path = next(
    Path(p)
    for p in ["/content/drive/MyDrive/AI/hipogpt", Path(__file__).parent]
    if Path(p).exists()
) / "saves"
saves_path.mkdir(exist_ok=True)


class Config:
    sample_only: str = True
    seed = 0

    # Dataset
    input_path = datasets_path / "murakami/murakami-1000lines.txt"
    vocab_size: int = 30_000
    context_len: int = 32

    # Network
    emb_dim: int = 32
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.1

    # Optimization
    batch_size: int = 32
    learning_rate: float = 1e-3  # 5e-4
    weight_decay: float = 0.01
    dataloader_num_workers: int = 0
    max_steps: int = 100_000
    checkpoint_steps: int = 100


torch.manual_seed(Config.seed)
torch.cuda.manual_seed_all(Config.seed)

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = ByteLevelBPETokenizer()
tok_prefix = Config.input_path.with_name(Config.input_path.stem + "-tokenizer")
tok_files = [f"{tok_prefix}-vocab.json", f"{tok_prefix}-merges.txt"]
try:
    tokenizer = tokenizer.from_file(*tok_files)
except Exception as e:
    if "No such file or directory" in str(e):
        print(f"Training tokenizer from {Config.input_path}")
        tokenizer.train(files=str(Config.input_path))
        tokenizer.save_model(str(tok_prefix.parent), tok_prefix.name)
        print(f"Saving to {tok_files}")
    else:
        raise


@dataclass
class InputDataClass:
    x: torch.Tensor
    y: Optional[torch.Tensor]


class TransformerDataset(Dataset):
    def __init__(self, text: str, context_len: int):
        super().__init__()
        self.context_len = context_len
        self.data = torch.tensor(tokenizer.encode(text).ids, dtype=torch.long)

    def __getitem__(self, index) -> InputDataClass:
        x = self.data[index : index + self.context_len]
        y = self.data[index + 1 : index + self.context_len + 1]
        return InputDataClass(x, y)

    def __len__(self):
        return len(self.data) - self.context_len - 1


with Config.input_path.open(encoding="utf-8") as f:
    dataset = TransformerDataset(f.read(), Config.context_len)

test_n = min(1000, int(len(dataset) * 0.1))
train_set, test_set = random_split(dataset, [len(dataset) - test_n, test_n])

dataloader = DataLoader(
    train_set,
    batch_size=Config.batch_size,
    sampler=torch.utils.data.RandomSampler(
        train_set, replacement=True, num_samples=int(1e10)
    ),
    num_workers=Config.dataloader_num_workers,
)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, emb_dim: int, ctx_len: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(
            emb_dim,
            n_heads,
            dropout,
            batch_first=True,
            device=device,
        )
        self.register_buffer(  # Where we are not allowed to attend
            "attn_mask", torch.ones((ctx_len, ctx_len)).tril() == 0
        )

        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        x = x + self.norm1(self.attn(x, x, x, attn_mask=self.attn_mask)[0])
        x = x + self.norm2(self.mlp(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        ctx_len: int,
        emb_dim: int,
        n_layers: int,
        n_heads: int,
        dropout=0.1,
    ):
        super().__init__()
        self.ctx_len = ctx_len
        self.te = nn.Embedding(vocab_size, emb_dim)
        self.pe = nn.Embedding(ctx_len, emb_dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(
            *[Block(emb_dim, ctx_len, n_heads, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(emb_dim)
        # self.transformer = nn.Transformer(
        #     d_model=emb_dim,
        #     nhead=n_heads,
        #     num_encoder_layers=0,
        #     num_decoder_layers=n_layers,
        #     dim_feedforward=emb_dim * 4,
        #     dropout=dropout,
        #     activation="gelu",
        #     batch_first=True,
        #     norm_first=True,
        # )
        self.lm_head = nn.Linear(emb_dim, vocab_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        # ignored, but needed to be in signature for the huggingface trainer
        return_loss: bool = True,  
    ):
        b, t = x.shape
        if t >= self.ctx_len:
            x = x[:, -self.ctx_len:]

        pos = torch.arange(0, t, dtype=torch.long, device=x.device).unsqueeze(0)
        x = self.te(x) + self.pe(pos)
        x = self.drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.lm_head(x)

        if y is not None:
            loss = nn.functional.cross_entropy(
                x.view(b * t, -1), y.view(b * t), ignore_index=-1
            )
            return loss, x
        return x

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_len: int = 100,
        temperature: float = 1.0,
        top_k: int = None,
    ):
        model.eval()  # turn on eval mode
        x = prompt
        for _ in range(max_len):
            logits = self(x[:, -self.ctx_len:])
            logits = logits[:, -1, :]  # take last token
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = logits.softmax(dim=-1)
            x_next = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, x_next), dim=1)
        return x


model = GPT(
    vocab_size=Config.vocab_size,
    ctx_len=Config.context_len,
    emb_dim=Config.emb_dim,
    n_layers=Config.n_layers,
    n_heads=Config.n_heads,
    dropout=Config.dropout,
).to(device)
print(f"Created model with {sum(p.numel() for p in model.parameters()):,} parameters")

prompt = dataset[0].x.unsqueeze(0).to(device)


def _callback(state: TrainerState, metrics: Dict[str, float] = None, **kwargs):
    if metrics:
        print(f'Eval loss so far: {metrics["eval_loss"]:.4f}')
    if state.best_metric:
        print(f"Best loss so far: {state.best_metric:.4f}")
    x_init = prompt.view(1, -1)
    x_sampled = model.generate(x_init, max_len=100).to("cpu")
    print("Generated text example:")
    print(tokenizer.decode(x_sampled[0].tolist()))


class PrintSampleCallback(TrainerCallback):
    def on_train_begin(self, _, state: TrainerState, *args, **kwargs):
        _callback(state, **kwargs)

    def on_evaluate(self, _, state: TrainerState, *args, **kwargs):
        _callback(state, **kwargs)


# Initialize Trainer
trainer = Trainer(
    model=model,
    optimizers=(
        torch.optim.AdamW(
            model.parameters(),
            lr=Config.learning_rate,
            weight_decay=Config.weight_decay,
        ),
        None,
    ),
    args=TrainingArguments(
        output_dir=str(saves_path),
        per_device_train_batch_size=Config.batch_size,
        per_device_eval_batch_size=100,
        data_seed=Config.seed,
        max_steps=Config.max_steps,
        dataloader_pin_memory=True,
        dataloader_num_workers=Config.dataloader_num_workers,
        load_best_model_at_end=True,
        evaluation_strategy="steps",
        logging_steps=Config.checkpoint_steps,
        eval_steps=Config.checkpoint_steps,
        save_steps=Config.checkpoint_steps,
    ),
    train_dataset=train_set,
    eval_dataset=test_set,
    callbacks=[PrintSampleCallback],
)

try:
    try:
        train_result = trainer.train(resume_from_checkpoint=True)
    except ValueError as e:
        if "No valid checkpoint found" in str(e):
            print(f"Checkpoint not found in {saves_path}, training from scratch")
            train_result = trainer.train()
        else:
            raise
except KeyboardInterrupt:
    print("Training interrupted by user")

print("Final result:")
_callback(trainer.state)
