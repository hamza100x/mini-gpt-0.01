# Mini GPT 0.01 (Essay-Only)

A very simple GPT-like toy model that trains **only on your own essay** and then generates predictions.

This version is intentionally tiny:
- Character-level model
- Bigram transitions (`current_char -> next_char`)
- No external dependencies

## 1) Add your essay

Put your full essay text in:

- `essay.txt`

## 2) Train

```bash
python mini_gpt.py train --input essay.txt --model model.json
```

## 3) Predict / generate

```bash
python mini_gpt.py predict --model model.json --prompt "In this essay" --length 300 --temperature 0.9
```

## 4) Chat (interactive query)

```bash
python mini_gpt.py chat --model model.json --length 180 --temperature 0.7
```

Or simply run this (defaults to chat):

```bash
python mini_gpt.py
```

Then type your query:

- `You: What is the main idea?`
- `You: summarize this`
- `You: quit` to exit

## Useful options

- `--temperature 0`: greedy (most likely next char)
- `--temperature 0.7`: more stable
- `--temperature 1.2`: more creative/random
- `--seed 42`: reproducible output

## Notes

- If your essay is short, output quality will be limited.
- This is a learning project, not a production GPT.
