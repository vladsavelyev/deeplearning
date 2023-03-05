import os
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import Dataset

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint


DRIVE_PATH = Path("/Users/vlad/googledrive")


tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")
model = AutoModelForCausalLM.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")


class TransformerDataset(Dataset):
    def __init__(self, token_ids: np.memmap, n_ctx: int):
        self.token_ids = token_ids
        self.n_ctx = n_ctx

    def __getitem__(self, idx):
        x = torch.LongTensor(self.token_ids[idx:idx + self.n_ctx])
        y = torch.LongTensor(self.token_ids[idx + 1:idx + 1 + self.n_ctx])
        return {"input_ids": x, "labels": y}

    def __len__(self):
        return len(self.token_ids) - self.n_ctx

    @staticmethod
    def load(path, tokenizer, n_ctx: int) -> 'TransformerDataset':
        save_path = path.with_suffix(".token_ids.pt")
        if save_path.exists():
            print(f"Loading dataset from {save_path}")
            ids = torch.load(str(save_path))
        else:
            with open(path, "r") as f:
                text = f.read()
                print(f"Characters in text: {len(text):,}")
            ids = tokenizer(text, return_tensors="pt")['input_ids'].squeeze().long()
            eos = torch.tensor([tokenizer.eos_token_id]).long()
            ids = torch.concat((ids, eos))
            torch.save(ids, save_path)
        print(f"Dataset shape: {ids.shape}")
        return TransformerDataset(ids, n_ctx)


test_text_path = DRIVE_PATH / "AI" / "datasets" / "murakami" / "murakami_test.txt"
train_text_path = DRIVE_PATH / "AI" / "datasets" / "murakami" / "murakami_train.txt"
test_set = TransformerDataset.load(test_text_path, tokenizer, model.config.n_ctx)
train_set = TransformerDataset.load(train_text_path, tokenizer, model.config.n_ctx)


save_dir = DRIVE_PATH / "AI" / "pretgpt" / "murakami_rugpt3small"
save_dir.mkdir(exist_ok=True)
if last_checkpoint_dir := get_last_checkpoint(str(save_dir)):
    last_checkpoint_dir = Path(last_checkpoint_dir)
    print([t.name for t in last_checkpoint_dir.iterdir()])


os.environ['WANDB_API_KEY'] = '270e81630bd1fd3c78a355d1711966c75ce75bcc'
os.environ["WANDB_NOTEBOOK_NAME"] = "pretgpt"


def sample(num_seqs=10):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    for i, seq in enumerate(model.generate(
        max_length=10,
        top_p=0.95,
        num_return_sequences=num_seqs,
        do_sample=True, 
        top_k=50,
        pad_token_id=0,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )):
        print(i, tokenizer.decode(seq))


class MyCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        sample()


trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir=str(save_dir),
        report_to=['wandb'],
        evaluation_strategy="epoch",
        overwrite_output_dir=True,
        eval_steps=500,
        save_steps=500,
        save_total_limit=2,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        ignore_data_skip=True,
    ),
    train_dataset=train_set,
    callbacks=[MyCallback],
)
# trainer.train(resume_from_checkpoint=last_checkpoint_dir)
trainer.train()


