#!/usr/bin/env python3
"""
mini_gpt.py
A very small, character-level language model trainer/generator.

This is intentionally simple: it learns bigram character transitions
from your essay and then predicts/generates text based on that.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("Input text is empty. Add your essay content first.")
    return text


def build_vocab(text: str) -> List[str]:
    # Sort for reproducibility.
    return sorted(set(text))


def train_bigram_counts(text: str, vocab: Sequence[str]) -> List[List[int]]:
    idx: Dict[str, int] = {ch: i for i, ch in enumerate(vocab)}
    size = len(vocab)
    counts = [[0 for _ in range(size)] for _ in range(size)]

    # Count transitions text[i] -> text[i+1]
    for left, right in zip(text[:-1], text[1:]):
        counts[idx[left]][idx[right]] += 1

    return counts


def choose_next_char(
    prev_char: str,
    vocab: Sequence[str],
    counts: List[List[int]],
    temperature: float = 1.0,
) -> str:
    idx: Dict[str, int] = {ch: i for i, ch in enumerate(vocab)}

    if prev_char not in idx:
        # If prompt ends with unknown char, start from random known char.
        return random.choice(list(vocab))

    row = counts[idx[prev_char]]

    # Add-one smoothing so unseen transitions still get a small chance.
    smoothed = [c + 1 for c in row]

    if temperature <= 0:
        # Greedy mode: choose max probability.
        best_i = max(range(len(smoothed)), key=lambda i: smoothed[i])
        return vocab[best_i]

    # Temperature scaling in count-space.
    scaled = [pow(v, 1.0 / temperature) for v in smoothed]
    total = sum(scaled)
    if total <= 0:
        return random.choice(list(vocab))

    pick = random.random() * total
    cumsum = 0.0
    for i, w in enumerate(scaled):
        cumsum += w
        if cumsum >= pick:
            return vocab[i]

    return vocab[-1]


def save_model(path: Path, vocab: Sequence[str], counts: List[List[int]]) -> None:
    payload = {
        "model_type": "char_bigram_v0.01",
        "vocab": list(vocab),
        "counts": counts,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load_model(path: Path) -> tuple[List[str], List[List[int]]]:
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    vocab = payload.get("vocab")
    counts = payload.get("counts")

    if not isinstance(vocab, list) or not isinstance(counts, list):
        raise ValueError("Invalid model format.")

    return vocab, counts


def train_command(input_path: Path, model_path: Path) -> None:
    text = read_text(input_path)
    vocab = build_vocab(text)
    counts = train_bigram_counts(text, vocab)
    save_model(model_path, vocab, counts)

    print("Training complete.")
    print(f"Input chars: {len(text)}")
    print(f"Unique chars (vocab): {len(vocab)}")
    print(f"Model saved to: {model_path}")


def predict_command(
    model_path: Path,
    prompt: str,
    length: int,
    temperature: float,
    seed: int | None,
) -> None:
    if seed is not None:
        random.seed(seed)

    vocab, counts = load_model(model_path)
    if not vocab:
        raise ValueError("Model vocab is empty.")

    print(generate_text(vocab, counts, prompt, length, temperature))


def generate_text(
    vocab: Sequence[str],
    counts: List[List[int]],
    prompt: str,
    length: int,
    temperature: float,
) -> str:
    if not prompt:
        prompt = random.choice(vocab)

    output = list(prompt)
    prev = output[-1]

    for _ in range(length):
        nxt = choose_next_char(prev, vocab, counts, temperature=temperature)
        output.append(nxt)
        prev = nxt

    return "".join(output)


def chat_command(
    model_path: Path,
    length: int,
    temperature: float,
    seed: int | None,
) -> None:
    if seed is not None:
        random.seed(seed)

    vocab, counts = load_model(model_path)
    if not vocab:
        raise ValueError("Model vocab is empty.")

    print("Mini GPT 0.01 chat mode")
    print("Type your query and press Enter. Type 'exit' or 'quit' to stop.")

    while True:
        try:
            query = input("You: ").strip()
        except EOFError:
            print("\nSession closed.")
            break
        except KeyboardInterrupt:
            print("\nSession interrupted.")
            break

        if not query:
            continue

        if query.lower() in {"exit", "quit"}:
            print("Bye.")
            break

        answer = generate_text(vocab, counts, query, length, temperature)
        print(f"Model: {answer}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mini GPT 0.01 (character bigram) trained only on your essay"
    )
    sub = parser.add_subparsers(dest="command")

    train = sub.add_parser("train", help="Train model from an essay text file")
    train.add_argument("--input", type=Path, default=Path("essay.txt"), help="Path to your essay")
    train.add_argument(
        "--model",
        type=Path,
        default=Path("model.json"),
        help="Path to save trained model",
    )

    predict = sub.add_parser("predict", help="Generate text from a prompt")
    predict.add_argument(
        "--model",
        type=Path,
        default=Path("model.json"),
        help="Path to trained model",
    )
    predict.add_argument("--prompt", type=str, default="", help="Starting text")
    predict.add_argument(
        "--length", type=int, default=200, help="How many new chars to generate"
    )
    predict.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Lower = more deterministic, higher = more random",
    )
    predict.add_argument("--seed", type=int, default=None, help="Random seed")

    chat = sub.add_parser("chat", help="Interactive query mode")
    chat.add_argument(
        "--model",
        type=Path,
        default=Path("model.json"),
        help="Path to trained model",
    )
    chat.add_argument(
        "--length", type=int, default=180, help="How many new chars to generate"
    )
    chat.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Lower = more deterministic, higher = more random",
    )
    chat.add_argument("--seed", type=int, default=None, help="Random seed")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "train":
            train_command(args.input, args.model)
        elif args.command == "predict":
            predict_command(
                model_path=args.model,
                prompt=args.prompt,
                length=args.length,
                temperature=args.temperature,
                seed=args.seed,
            )
        elif args.command == "chat":
            chat_command(
                model_path=args.model,
                length=args.length,
                temperature=args.temperature,
                seed=args.seed,
            )
        else:
            # Default behavior: start chat with standard settings.
            chat_command(
                model_path=Path("model.json"),
                length=180,
                temperature=0.7,
                seed=None,
            )
    except FileNotFoundError as exc:
        print(exc)
        print("Tip: run training first -> python mini_gpt.py train --input essay.txt --model model.json")


if __name__ == "__main__":
    main()
