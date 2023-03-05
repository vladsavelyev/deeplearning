from pathlib import Path
DRIVE_PATH = Path("/Users/vlad/googledrive")

from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")
model = AutoModelForCausalLM.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")

from transformers.trainer_utils import get_last_checkpoint
save_dir = DRIVE_PATH / "AI" / "pretgpt" / "murakami_rugpt3small"
save_dir.mkdir(exist_ok=True)
if last_checkpoint_dir := get_last_checkpoint(str(save_dir)):
    last_checkpoint_dir = Path(last_checkpoint_dir)
    print(last_checkpoint_dir)
    print('  '.join(t.name for t in last_checkpoint_dir.iterdir()))

from transformers.utils import WEIGHTS_NAME
import torch
state_dict = torch.load(str(last_checkpoint_dir / WEIGHTS_NAME), map_location="cpu")
model.load_state_dict(state_dict)    

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


def generate(**kwargs):
    seq = model.generate(
        max_length=500,
        num_return_sequences=1,
        pad_token_id=0,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        **kwargs,
    )[0]
    print(tokenizer.decode(seq))
    print()


print("My search:")
generate(
    do_sample=True, 
    top_p=0.95,
    top_k=50,
)

print("Greedy search:")
generate()

print("Nucleus sampling:")
generate(
    do_sample=True,
    top_p=0.05,
    top_k=0,
)

print("Beam search:")
generate(
    do_sample=False,
    num_beams=4,
)

print("Beam multinomial search:")
generate(
    do_sample=True,
    num_beams=4,
)

print("Contrastive search:")
generate(
    penalty_alpha=0.6, 
    top_k=4,
)
