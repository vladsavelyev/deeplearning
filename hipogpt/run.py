from typing import Dict, Optional
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from transformers import (
    Trainer,
    TrainingArguments,
    TrainerState,
    TrainerCallback,
)

if "get_dataset" not in globals():
    from hipogpt.data import get_dataset


class Config:
    seed = 42
    restart = True  # ignore existing saves

    # Dataset
    dataset = "murakami"

    # Network
    emb_dim: int = 128
    n_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.1

    # Optimization
    batch_size: int = 32
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    dataloader_num_workers: int = 0
    max_steps: int = 100_000
    checkpoint_steps: int = 500  # log, eval, save

    # Generation
    temperature = 0.8


drive_path = Path("/Users/vlad/MyDrive/AI")
dataset = get_dataset(Config.dataset, drive_path / "datasets")
saves_path = drive_path / "hipogpt" / "saves" / dataset.path.stem
saves_path.mkdir(exist_ok=True)

torch.manual_seed(Config.seed)
torch.cuda.manual_seed_all(Config.seed)

device = "cuda" if torch.cuda.is_available() else "cpu"

dataloader = DataLoader(
    dataset.train,
    batch_size=Config.batch_size,
    sampler=torch.utils.data.RandomSampler(
        dataset.train, replacement=True, num_samples=int(1e10)
    ),
    num_workers=Config.dataloader_num_workers,
)


# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)
#
#         pos_emb = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(
#             torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
#         )
#         pos_emb[:, 0::2] = torch.sin(position * div_term)
#         pos_emb[:, 1::2] = torch.cos(position * div_term)
#         pos_emb = pos_emb.unsqueeze(0).transpose(0, 1)
#         self.register_buffer("pos_emb", pos_emb)
#
#     def forward(self, x):
#         x = x + self.pos_emb[: x.shape[0], :]
#         return self.dropout(x)


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
        dropout: float,
    ):
        super().__init__()
        self.ctx_len = ctx_len
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(ctx_len, emb_dim)
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
        # if y is not None:
        #     print('Train examples:')
        #     for x1, y1 in zip(x.tolist(), y.tolist()):
        #         print(
        #             f'{dataset.tokenizer.decode(x1)} -> '
        #             f'{dataset.tokenizer.decode(y1)}'
        #         )
        b, t = x.shape
        if t > self.ctx_len:
            x = x[:, -self.ctx_len :]
        elif t < self.ctx_len:
            raise ValueError(
                f"Prompt is shorter than the context length: {t=} {self.ctx_len=}"
            )
        b, t = x.shape

        pos = torch.arange(0, t, dtype=torch.long, device=x.device).unsqueeze(0)
        x1 = self.token_emb(x)
        x2 = self.pos_emb(pos)
        x = x1 + x2
        x = self.drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.lm_head(x)

        if y is not None:
            loss = nn.functional.cross_entropy(
                x.view(b * t, -1),
                y.view(b * t),
                ignore_index=dataset.get_ignore_index(),
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
        stop_token: Optional[int] = None,
    ):
        model.eval()  # turn on eval mode
        x = prompt.to(device).view(1, -1)

        for _ in range(max_len):
            logits = self(x)
            logits = logits[:, -1, :]  # take last token
            if temperature > 0.0:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float("inf")
                probs = logits.softmax(dim=-1)
                x_next = torch.multinomial(probs, num_samples=1)
            else:
                x_next = logits.argmax(dim=-1)
            if x_next == stop_token:
                break
            x = torch.cat((x, x_next), dim=1)
        return x


model = GPT(
    vocab_size=dataset.vocab_size,
    ctx_len=dataset.get_context_len(),
    emb_dim=Config.emb_dim,
    n_layers=Config.n_layers,
    n_heads=Config.n_heads,
    dropout=Config.dropout,
).to(device)
print(f"Created model with {sum(p.numel() for p in model.parameters()):,} parameters")


def _callback(state: TrainerState, metrics: Dict[str, float] = None, **kwargs):
    if metrics:
        print(f'Eval loss so far: {metrics["eval_loss"]:.4f}')
    if state.best_metric:
        print(f"Best loss so far: {state.best_metric:.4f}")
    dataset.print_examples(model, temperature=Config.temperature)


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
    train_dataset=dataset.train,
    eval_dataset=dataset.test,
    callbacks=[PrintSampleCallback],
)

try:
    resume_from_checkpoint = not Config.restart and any(
        p.name.startswith("checkpoint-") for p in saves_path.iterdir()
    )
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
except KeyboardInterrupt:
    print("Training interrupted by user")

print("Final result:")
_callback(trainer.state)
