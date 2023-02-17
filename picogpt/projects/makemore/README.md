# Makemore character-level model

Takes a list of words as input, and generates more similar words. E.g. can be used to imagine novel baby names.

Usage: 

```sh
python makemore.py data/names.txt
```

Will save and reuse the prepared dataset as `data/names.txt.pt`, and the model as `runs/model.pt`.
