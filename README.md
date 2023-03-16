# deeplearning

Studying and playing with AI.

* [GTP to GPT](gtp_gpt): training a LM on guitar tablature.
* Path to GPT:
	* [karpathy](karpathy): following [Andrej Karpathy's course](https://github.com/karpathy/nn-zero-to-hero) from a primitive language model to GPT to make up names.
	* [picogpt](picogpt): one step from Karpathy's course to industry ML: using `torch.nn` instead of implementing models manually.
	* [hipogpt](hipogpt): another step further: using HuggingFace `transformers` library to train.
* [pretgpt](pretgpt): fine-tuning a pre-trained GPT2 to generate Murakami texts.
* [scribepod](scribepod): forking and playing with scribepod to generate a podcast about bioinformatics papers.
* [notionbot](notionbot/notionbot.py): using [llama-index](https://gpt-index.readthedocs.io/en/latest/index.html), [langchain](https://langchain.readthedocs.io/en/latest/), and SQL, build a bot that can answer questions about yourself based on a content of a Notion database without fine-tuning.