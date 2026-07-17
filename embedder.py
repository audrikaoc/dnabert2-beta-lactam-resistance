#!/usr/bin/env python3
"""Generate DNABERT-2 embeddings for DNA sequences.

The main function, ``embed_dataframe``, accepts a pandas DataFrame with a DNA
sequence column, runs the sequences through ``zhihan1996/DNABERT-2-117M`` using
PyTorch/transformers, mean-pools token embeddings into one 768-dimensional
vector per sequence, and optionally saves the final matrix as ``.npy``.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from transformers.models.bert.configuration_bert import BertConfig


MODEL_NAME = "zhihan1996/DNABERT-2-117M"
DEFAULT_SEQUENCE_COLUMN = "sequence"
EXPECTED_EMBEDDING_DIM = 768


def _choose_device(requested_device: Optional[str] = None) -> torch.device:
    """Pick an inference device, preferring accelerators when available."""
    if requested_device:
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _get_last_hidden_state(model_outputs) -> torch.Tensor:
    """Return token-level embeddings from either dict-like or tuple outputs."""
    if hasattr(model_outputs, "last_hidden_state"):
        return model_outputs.last_hidden_state
    if isinstance(model_outputs, dict) and "last_hidden_state" in model_outputs:
        return model_outputs["last_hidden_state"]
    return model_outputs[0]


def _patch_dnabert_alibi_builder() -> bool:
    """Patch cached DNABERT-2 remote code for newer transformers meta init.

    Newer transformers versions instantiate remote models under a meta-device
    context. DNABERT-2's cached ALiBi builder can mix meta and CPU tensors in
    that context, so this replaces the builder with the same logic while
    creating all temporary tensors on one device.
    """
    remote_modules = [
        module
        for module in sys.modules.values()
        if getattr(module, "__name__", "").endswith(".bert_layers")
        and hasattr(module, "BertEncoder")
    ]
    if not remote_modules:
        return False

    bert_layers = remote_modules[-1]
    bert_layers.flash_attn_qkvpacked_func = None

    def rebuild_alibi_tensor(self, size: int, device: Optional[torch.device | str] = None):
        if device is None:
            device = self.alibi.device

        n_heads = self.num_attention_heads

        def get_alibi_head_slopes(head_count: int) -> List[float]:
            def get_slopes_power_of_2(power_head_count: int) -> List[float]:
                start = 2 ** (-2 ** -(math.log2(power_head_count) - 3))
                ratio = start
                return [start * ratio**i for i in range(power_head_count)]

            if math.log2(head_count).is_integer():
                return get_slopes_power_of_2(head_count)

            closest_power_of_2 = 2 ** math.floor(math.log2(head_count))
            slopes_a = get_slopes_power_of_2(closest_power_of_2)
            slopes_b = get_alibi_head_slopes(2 * closest_power_of_2)
            return slopes_a + slopes_b[0::2][: head_count - closest_power_of_2]

        context_position = torch.arange(size, device=device)[:, None]
        memory_position = torch.arange(size, device=device)[None, :]
        relative_position = torch.abs(memory_position - context_position)
        relative_position = relative_position.unsqueeze(0).expand(n_heads, -1, -1)
        slopes = torch.tensor(
            get_alibi_head_slopes(n_heads),
            device=device,
            dtype=relative_position.dtype,
        )

        self._current_alibi_size = size
        self.alibi = (slopes.unsqueeze(1).unsqueeze(1) * -relative_position).unsqueeze(0)

    bert_layers.BertEncoder.rebuild_alibi_tensor = rebuild_alibi_tensor
    return True


def load_dnabert_model(model_name: str = MODEL_NAME) -> torch.nn.Module:
    """Load DNABERT-2 with a fallback for newer transformers versions."""
    try:
        return AutoModel.from_pretrained(model_name, trust_remote_code=True)
    except AttributeError as exc:
        if "pad_token_id" not in str(exc):
            raise

    if not _patch_dnabert_alibi_builder():
        raise RuntimeError("DNABERT-2 remote module was not available to patch")

    config = BertConfig.from_pretrained(model_name)
    config.pad_token_id = 0
    config.bos_token_id = 0
    config.eos_token_id = 1
    model = AutoModel.from_pretrained(
        model_name,
        config=config,
        trust_remote_code=True,
    )
    if hasattr(model, "encoder") and hasattr(model.encoder, "rebuild_alibi_tensor"):
        model.encoder.rebuild_alibi_tensor(config.alibi_starting_size, device="cpu")
    return model


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token embeddings while ignoring padding tokens."""
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed_embeddings = (last_hidden_state * mask).sum(dim=1)
    token_counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed_embeddings / token_counts


def embed_dataframe(
    df: pd.DataFrame,
    sequence_column: str = DEFAULT_SEQUENCE_COLUMN,
    model_name: str = MODEL_NAME,
    batch_size: int = 8,
    max_length: int = 512,
    output_path: Optional[str | Path] = None,
    device: Optional[str] = None,
) -> np.ndarray:
    """Embed DNA sequences from a DataFrame and return an ``(n, 768)`` matrix.

    Parameters
    ----------
    df:
        DataFrame containing one DNA sequence per row.
    sequence_column:
        Name of the DataFrame column containing DNA strings.
    model_name:
        Hugging Face model identifier or local model directory.
    batch_size:
        Number of sequences to process at once. Keep this modest on laptops.
    max_length:
        Maximum tokenizer length. Longer sequences are truncated.
    output_path:
        Optional ``.npy`` path where the embedding matrix will be saved.
    device:
        Optional PyTorch device string, such as ``"cpu"``, ``"cuda"``, or
        ``"mps"``. If omitted, the script picks the best available device.
    """
    if sequence_column not in df.columns:
        raise KeyError(f"DataFrame must contain a '{sequence_column}' column")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if max_length < 1:
        raise ValueError("max_length must be at least 1")

    sequences = df[sequence_column].dropna().astype(str).str.upper().tolist()
    if not sequences:
        raise ValueError("No non-empty sequences were found in the input DataFrame")

    torch_device = _choose_device(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = load_dnabert_model(model_name)
    model.to(torch_device)
    model.eval()

    batches: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch_sequences = sequences[start : start + batch_size]
            tokenized = tokenizer(
                batch_sequences,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            tokenized = {key: value.to(torch_device) for key, value in tokenized.items()}

            outputs = model(**tokenized)
            last_hidden_state = _get_last_hidden_state(outputs)
            pooled = mean_pooling(last_hidden_state, tokenized["attention_mask"])
            batches.append(pooled.detach().cpu().numpy().astype(np.float32))

    embedding_matrix = np.vstack(batches)
    if embedding_matrix.shape[1] != EXPECTED_EMBEDDING_DIM:
        raise ValueError(
            f"Expected {EXPECTED_EMBEDDING_DIM}-dimensional embeddings, "
            f"but got shape {embedding_matrix.shape}"
        )

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_file, embedding_matrix)

    return embedding_matrix


def main() -> None:
    """Read a CSV of sequences, generate embeddings, and save a NumPy file."""
    parser = argparse.ArgumentParser(
        description="Generate mean-pooled DNABERT-2 embeddings from a sequence CSV."
    )
    parser.add_argument("input_csv", help="CSV file containing a DNA sequence column")
    parser.add_argument(
        "--output",
        "-o",
        default="embeddings.npy",
        help="Output path for the NumPy .npy embedding matrix",
    )
    parser.add_argument(
        "--sequence-column",
        default=DEFAULT_SEQUENCE_COLUMN,
        help="Column containing DNA sequences",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Inference batch size; lower this if memory is tight",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum tokenized length; longer sequences are truncated",
    )
    parser.add_argument(
        "--device",
        default=None,
        help='Optional PyTorch device override, e.g. "cpu", "cuda", or "mps"',
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    embeddings = embed_dataframe(
        df=df,
        sequence_column=args.sequence_column,
        batch_size=args.batch_size,
        max_length=args.max_length,
        output_path=args.output,
        device=args.device,
    )

    print(f"Generated embedding matrix with shape {embeddings.shape}")
    print(f"Saved embeddings to {args.output}")


if __name__ == "__main__":
    main()
