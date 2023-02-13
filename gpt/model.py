"""
Transformer Language Model (*exactly* as used in GPT-2)
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT
    repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU)
    Paper: https://arxiv.org/abs/1606.08415
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

        # Causal mask to ensure that attention is only applied to the left in the
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
    """
    Transformer Language Model, exactly as seen in GPT-2
    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        emb_dim: int,
        n_layers: int,
        n_heads: int,
    ):
        super().__init__()
        self.block_size = block_size
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(vocab_size, emb_dim),
                wpe=nn.Embedding(vocab_size, emb_dim),
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
        print(f"Transformer parameters: {n_params}")

    def forward(self, x, targets=None) -> tuple[torch.Tensor, torch.Tensor | None]:
        device = x.device
        b, t = x.shape
        assert t <= self.block_size, (
            f"Cannot forward sequence of length {t}, "
            f"block size is only {self.block_size}"
        )
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # (1, t)

        # forward the GTP model
        tok_emb = self.transformer.wte(x)  # (b, t, c)
        pos_emb = self.transformer.wpe(pos)  # (1, t, c)
        x = tok_emb + pos_emb  # (b, t, c)
        for block in self.transformer.blocks:
            x = block(x)
        x = self.transformer.lnorm(x)

        if targets is not None:
            logits = self.lm_head(x)
            b, t, vocab_size = logits.shape
            loss = F.cross_entropy(
                logits.view(b * t, vocab_size),  # (b*t, vocab_size)
                targets.view(b * t),  # (b*t)
                ignore_index=-1,
            )
        else:
            loss = None
            # Inference-time mini-optimization: only forward the lm_head on the 
            # very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to 
            # preserve the time dim
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """
        Take a conditioning sequence of indices x (LongTensor of shape (b, t))
        and complete the sequence max_new_tokens times, feeding the predictions back
        into the model each time. Most likely you'll want to make sure to be in
        model.eval() mode of operation for this.
        """
        block_size = self.block_size
        for _ in range(max_new_tokens):
            # If the sequence context is growing too long we must crop it at block_size
            x = x if x.shape[1] <= block_size else x[:, -block_size:]
    
            # Forward the model to get the logits for the index in the sequence
            logits, _ = self(x)
    
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
