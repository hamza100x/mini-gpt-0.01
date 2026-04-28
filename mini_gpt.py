#!/usr/bin/env python3
"""
mini_gpt.py
A tiny neural character model trained only on your essay.

This version uses a small feed-forward neural network over a fixed
character window so the outputs are less random than the earlier bigram toy.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


MODEL_TYPE = "char_mlp_v0.02"
DEFAULT_BLOCK_SIZE = 8
DEFAULT_HIDDEN_SIZE = 192
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 128
DEFAULT_LEARNING_RATE = 0.05
DEFAULT_TOP_K = 8


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("Input text is empty. Add your essay content first.")
    return text


def build_vocab(text: str) -> List[str]:
    return sorted(set(text))


def choose_pad_char(vocab: Sequence[str]) -> str:
    return " " if " " in vocab else vocab[0]


def make_dataset(
    text: str,
    vocab: Sequence[str],
    block_size: int,
) -> Tuple[np.ndarray, np.ndarray, str]:
    char_to_idx: Dict[str, int] = {ch: i for i, ch in enumerate(vocab)}
    pad_char = choose_pad_char(vocab)
    pad_idx = char_to_idx[pad_char]
    ids = [char_to_idx.get(ch, pad_idx) for ch in text]
    padded = [pad_idx] * block_size + ids

    x = np.empty((len(ids), block_size), dtype=np.int64)
    y = np.asarray(ids, dtype=np.int64)
    for i in range(len(ids)):
        x[i] = padded[i : i + block_size]

    return x, y, pad_char


def init_model(
    vocab_size: int,
    block_size: int,
    hidden_size: int,
    seed: int | None,
) -> dict:
    rng = np.random.default_rng(seed)
    scale = 0.02
    return {
        "W1": rng.normal(0.0, scale, (block_size * vocab_size, hidden_size)).astype(np.float32),
        "b1": np.zeros(hidden_size, dtype=np.float32),
        "W2": rng.normal(0.0, scale, (hidden_size, vocab_size)).astype(np.float32),
        "b2": np.zeros(vocab_size, dtype=np.float32),
    }


def one_hot_flat(x: np.ndarray, vocab_size: int) -> np.ndarray:
    return np.eye(vocab_size, dtype=np.float32)[x].reshape(x.shape[0], -1)


def forward(model: dict, x_flat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h_pre = x_flat @ model["W1"] + model["b1"]
    h = np.tanh(h_pre)
    logits = h @ model["W2"] + model["b2"]
    return h, logits


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def apply_temperature(probs: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    """Apply temperature scaling and sample from distribution."""
    if temperature <= 0:
        return np.argmax(probs)
    
    logits = np.log(probs + 1e-12) / temperature
    logits -= np.max(logits)
    adjusted = np.exp(logits)
    return rng.choice(len(adjusted), p=adjusted / np.sum(adjusted))


def train_step(model: dict, x_batch: np.ndarray, y_batch: np.ndarray, vocab_size: int, learning_rate: float) -> float:
    x_flat = one_hot_flat(x_batch, vocab_size)
    h, logits = forward(model, x_flat)
    probs = softmax(logits)
    batch_size = x_batch.shape[0]

    loss = -np.log(probs[np.arange(batch_size), y_batch] + 1e-12).mean()

    dlogits = probs.copy()
    dlogits[np.arange(batch_size), y_batch] -= 1.0
    dlogits /= batch_size

    dW2 = h.T @ dlogits
    db2 = dlogits.sum(axis=0)
    dh = dlogits @ model["W2"].T
    dh_pre = dh * (1.0 - h * h)
    dW1 = x_flat.T @ dh_pre
    db1 = dh_pre.sum(axis=0)

    model["W1"] -= learning_rate * dW1.astype(np.float32)
    model["b1"] -= learning_rate * db1.astype(np.float32)
    model["W2"] -= learning_rate * dW2.astype(np.float32)
    model["b2"] -= learning_rate * db2.astype(np.float32)

    return float(loss)


def train_neural_model(
    text: str,
    vocab: Sequence[str],
    block_size: int,
    hidden_size: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int | None,
) -> tuple[dict, str]:
    x, y, pad_char = make_dataset(text, vocab, block_size)
    vocab_size = len(vocab)
    model = init_model(vocab_size, block_size, hidden_size, seed)
    rng = np.random.default_rng(seed)

    for epoch in range(epochs):
        order = rng.permutation(len(x))
        x_shuffled = x[order]
        y_shuffled = y[order]
        total_loss = 0.0

        for start in range(0, len(x_shuffled), batch_size):
            end = start + batch_size
            x_batch = x_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            loss = train_step(model, x_batch, y_batch, vocab_size, learning_rate)
            total_loss += loss * len(x_batch)

        avg_loss = total_loss / len(x_shuffled)
        if epoch == 0 or (epoch + 1) % max(1, epochs // 5) == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f}")

    return model, pad_char


def save_model(
    path: Path,
    vocab: Sequence[str],
    block_size: int,
    hidden_size: int,
    pad_char: str,
    model: dict,
) -> None:
    np.savez_compressed(
        path,
        model_type=np.array(MODEL_TYPE),
        vocab=np.array(vocab),
        block_size=np.array(block_size, dtype=np.int64),
        hidden_size=np.array(hidden_size, dtype=np.int64),
        pad_char=np.array(pad_char),
        W1=model["W1"],
        b1=model["b1"],
        W2=model["W2"],
        b2=model["b2"],
    )


def load_model(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    data = np.load(path, allow_pickle=False)
    model_type = str(data["model_type"].item())
    if model_type != MODEL_TYPE:
        raise ValueError(f"Unsupported model type: {model_type}")

    return {
        "vocab": data["vocab"].tolist(),
        "block_size": int(data["block_size"].item()),
        "hidden_size": int(data["hidden_size"].item()),
        "pad_char": str(data["pad_char"].item()),
        "W1": data["W1"],
        "b1": data["b1"],
        "W2": data["W2"],
        "b2": data["b2"],
    }


def predict_next_probs(model: dict, context_ids: np.ndarray, vocab_size: int) -> np.ndarray:
    x_flat = one_hot_flat(context_ids[np.newaxis, :], vocab_size)
    _, logits = forward(model, x_flat)
    return softmax(logits)[0]


def sample_from_probs(
    probs: np.ndarray,
    temperature: float,
    rng: np.random.Generator,
    top_k: int,
) -> int:
    if top_k > 0 and top_k < len(probs):
        top_indices = np.argpartition(probs, -top_k)[-top_k:]
        probs = probs[top_indices]
        probs = probs / np.sum(probs)
        return top_indices[apply_temperature(probs, temperature, rng)]
    
    return apply_temperature(probs, temperature, rng)


def generate_text(
    model_bundle: dict,
    prompt: str,
    length: int,
    temperature: float,
    rng: np.random.Generator,
    top_k: int,
) -> str:
    vocab = model_bundle["vocab"]
    char_to_idx = {ch: i for i, ch in enumerate(vocab)}
    pad_char = model_bundle["pad_char"]
    pad_idx = char_to_idx[pad_char]
    block_size = model_bundle["block_size"]
    vocab_size = len(vocab)

    if not prompt:
        prompt = vocab[rng.integers(len(vocab))]

    output = list(prompt)

    for _ in range(length):
        context = output[-block_size:]
        if len(context) < block_size:
            context = [pad_char] * (block_size - len(context)) + context
        context_ids = np.array([char_to_idx.get(ch, pad_idx) for ch in context], dtype=np.int64)
        probs = predict_next_probs(model_bundle, context_ids, vocab_size)
        next_idx = sample_from_probs(probs, temperature, rng, top_k=top_k)
        output.append(vocab[next_idx])

    return "".join(output)


def train_command(
    input_path: Path,
    model_path: Path,
    epochs: int,
    hidden_size: int,
    block_size: int,
    batch_size: int,
    learning_rate: float,
    seed: int | None,
) -> None:
    text = read_text(input_path)
    vocab = build_vocab(text)
    model, pad_char = train_neural_model(
        text=text,
        vocab=vocab,
        block_size=block_size,
        hidden_size=hidden_size,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    )
    save_model(model_path, vocab, block_size, hidden_size, pad_char, model)

    print("Training complete.")
    print(f"Input chars: {len(text)}")
    print(f"Unique chars (vocab): {len(vocab)}")
    print(f"Window size: {block_size}")
    print(f"Hidden size: {hidden_size}")
    print(f"Model saved to: {model_path}")


def predict_command(
    model_path: Path,
    prompt: str,
    length: int,
    temperature: float,
    seed: int | None,
    top_k: int,
) -> None:
    rng = np.random.default_rng(seed)
    model_bundle = load_model(model_path)
    print(generate_text(model_bundle, prompt, length, temperature, rng, top_k))


def chat_command(
    model_path: Path,
    length: int,
    temperature: float,
    seed: int | None,
    top_k: int,
) -> None:
    rng = np.random.default_rng(seed)
    model_bundle = load_model(model_path)

    print("Mini GPT 0.01 chat mode")
    print("Type your query and press Enter. Type 'exit' or 'quit' to stop.")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession closed.")
            break

        if not query or query.lower() in {"exit", "quit"}:
            if query.lower() in {"exit", "quit"}:
                print("Bye.")
            break

        answer = generate_text(model_bundle, query, length, temperature, rng, top_k)
        print(f"Model: {answer}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mini GPT 0.01 (small neural character model) trained only on your essay"
    )
    sub = parser.add_subparsers(dest="command")

    train = sub.add_parser("train", help="Train model from an essay text file")
    train.add_argument("--input", type=Path, default=Path("essay.txt"), help="Path to your essay")
    train.add_argument(
        "--model",
        type=Path,
        default=Path("model.npz"),
        help="Path to save trained model",
    )
    train.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Training epochs")
    train.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE, help="Hidden layer size")
    train.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE, help="Context window size")
    train.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Mini-batch size")
    train.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Gradient descent learning rate",
    )
    train.add_argument("--seed", type=int, default=42, help="Random seed")

    predict = sub.add_parser("predict", help="Generate text from a prompt")
    predict.add_argument(
        "--model",
        type=Path,
        default=Path("model.npz"),
        help="Path to trained model",
    )
    predict.add_argument("--prompt", type=str, default="", help="Starting text")
    predict.add_argument("--length", type=int, default=200, help="How many new chars to generate")
    predict.add_argument(
        "--temperature",
        type=float,
        default=0.45,
        help="Lower = more deterministic, higher = more random",
    )
    predict.add_argument("--seed", type=int, default=None, help="Random seed")
    predict.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Sample only from the top K next chars")

    chat = sub.add_parser("chat", help="Interactive query mode")
    chat.add_argument(
        "--model",
        type=Path,
        default=Path("model.npz"),
        help="Path to trained model",
    )
    chat.add_argument("--length", type=int, default=180, help="How many new chars to generate")
    chat.add_argument(
        "--temperature",
        type=float,
        default=0.45,
        help="Lower = more deterministic, higher = more random",
    )
    chat.add_argument("--seed", type=int, default=None, help="Random seed")
    chat.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Sample only from the top K next chars")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        args.command = "chat"
        args.model = Path("model.npz")
        args.length = 180
        args.temperature = 0.45
        args.seed = None
        args.top_k = DEFAULT_TOP_K

    try:
        if args.command == "train":
            train_command(
                input_path=args.input,
                model_path=args.model,
                epochs=args.epochs,
                hidden_size=args.hidden_size,
                block_size=args.block_size,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                seed=args.seed,
            )
        elif args.command == "predict":
            predict_command(
                model_path=args.model,
                prompt=args.prompt,
                length=args.length,
                temperature=args.temperature,
                seed=args.seed,
                top_k=args.top_k,
            )
        elif args.command == "chat":
            chat_command(
                model_path=args.model,
                length=args.length,
                temperature=args.temperature,
                seed=args.seed,
                top_k=args.top_k,
            )
    except FileNotFoundError as exc:
        print(exc)
        print("Tip: run training first -> python mini_gpt.py train --input essay.txt --model model.npz")


if __name__ == "__main__":
    main()
