"""Run text through a model with a fitted Jacobian lens and show the lens
output at every token position and layer.

Output formats:
  text  (default)  Human-readable table on stdout (or --output file).
  json             Structured JSON with per-position, per-layer data.
  csv              Flat CSV: one row per (position, layer, rank).

Examples::

    # Human-readable table for a single prompt
    python scripts/infer.py Qwen/Qwen3.5-0.8B lenses/qwen.jlens.pt \\
        --prompt "The capital of France is"

    # JSON output with top-10 predictions
    python scripts/infer.py Qwen/Qwen3.5-0.8B lenses/qwen.jlens.pt \\
        --prompt "Hello world" --format json --top_k 10

    # Pipe text from a file
    cat story.txt | python scripts/infer.py Qwen/Qwen3.5-0.8B lenses/qwen.jlens.pt

    # Save to a file
    python scripts/infer.py Qwen/Qwen3.5-0.8B lenses/qwen.jlens.pt \\
        --prompt "Hello" --format json --output results.json
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import sys

import torch
import transformers

import jlens

logger = logging.getLogger("jlens")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    """Convert logits [n_pos, vocab] to probabilities, with float32 cast."""
    return torch.softmax(logits.float(), dim=-1)


def _extract_data(lens_logits, model_logits, input_ids, tokenizer, top_k: int):
    """Convert raw lens/model logits into per-position structured data.

    Returns:
        pos_data: list of dicts with keys: position, token, token_id,
                  lens_layers, model_topk
    """
    probs = _logits_to_probs(model_logits)  # [n_pos, vocab]
    model_topk_idx = probs.topk(top_k, dim=-1).indices  # [n_pos, top_k]
    model_topk_probs = probs.gather(-1, model_topk_idx)

    n_pos = input_ids.shape[1]
    token_ids = input_ids[0].tolist()
    pos_data = []

    for pos in range(n_pos):
        entry = {
            "position": pos,
            "token_id": token_ids[pos],
            "token": tokenizer.decode([token_ids[pos]], clean_up_tokenization_spaces=False),
            "lens_layers": {},
            "model_topk": [],
        }

        for k in range(top_k):
            tid = int(model_topk_idx[pos, k])
            entry["model_topk"].append({
                "token_id": tid,
                "token": tokenizer.decode([tid], clean_up_tokenization_spaces=False),
                "probability": float(model_topk_probs[pos, k]),
            })

        for layer in sorted(lens_logits):
            layer_logits = lens_logits[layer]  # [n_pos, vocab]
            layer_probs = _logits_to_probs(layer_logits)
            topk_idx = layer_probs[pos].topk(top_k).indices
            topk_probs = layer_probs[pos].gather(-1, topk_idx)

            layer_data = []
            for k in range(top_k):
                tid = int(topk_idx[k])
                layer_data.append({
                    "token_id": tid,
                    "token": tokenizer.decode([tid], clean_up_tokenization_spaces=False),
                    "probability": float(topk_probs[k]),
                })
            entry["lens_layers"][str(layer)] = layer_data

        pos_data.append(entry)

    return pos_data


# --------------------------------------------------------------------------- #
# Output formatters
# --------------------------------------------------------------------------- #


def _format_text(
    pos_data: list,
    layers: list[int],
    top_k: int,
    n_layers: int,
) -> str:
    """Render human-readable table. Returns a string."""

    lines: list[str] = []

    # Compute column widths
    col_width = max(4, top_k * 6)
    token_width = max(5, max(len(e["token"]) for e in pos_data) + 2)

    # Header
    header = "pos".ljust(5) + "token".ljust(token_width)
    for layer in layers:
        label = f"L{layer}" if layer < n_layers - 1 else "model"
        header += label.center(col_width)
    header += "  model pred"
    lines.append(header)
    lines.append("-" * len(header))

    # Rows
    for entry in pos_data:
        pos = entry["position"]
        token = entry["token"]
        row = str(pos).ljust(5) + repr(token).ljust(token_width)

        for layer in layers:
            layer_key = str(layer)
            layer_items = entry["lens_layers"][layer_key]
            parts = []
            for item in layer_items:
                tok_str = item["token"].strip()
                if not tok_str:
                    tok_str = repr(item["token"])
                prob = item["probability"]
                parts.append(f"{tok_str} {prob:.2f}")
            cell = " ".join(parts)
            row += cell.ljust(col_width)

        model_pred = entry["model_topk"][0] if entry["model_topk"] else None
        if model_pred:
            pred_token = model_pred["token"].strip() or repr(model_pred["token"])
            pred_prob = model_pred["probability"]
            row += f"  {pred_token} {pred_prob:.3f}"

        lines.append(row)

    # Position-level stats
    lines.append("")
    lines.append("Per-position model prediction (argmax from final layer):")
    lines.append("-" * 60)
    for entry in pos_data:
        pos = entry["position"]
        token = entry["token"]
        pred = entry["model_topk"][0]
        pred_token = pred["token"].strip() or repr(pred["token"])
        lines.append(
            f"  pos {pos:>4d}: {repr(token):<20s} -> {pred_token} ({pred['probability']:.4f})"
        )

    return "\n".join(lines)


def _format_json(pos_data: list, meta: dict) -> str:
    """JSON output."""
    return json.dumps({"metadata": meta, "positions": pos_data}, indent=2, ensure_ascii=False)


def _format_csv(pos_data: list, n_layers: int, top_k: int) -> str:
    """CSV: one row per (position, layer, rank)."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "position", "token", "token_id", "layer", "layer_type",
        "rank", "predicted_token", "predicted_token_id", "probability",
    ])
    for entry in pos_data:
        pos = entry["position"]
        token = entry["token"]
        tid = entry["token_id"]
        for layer_key, layer_items in entry["lens_layers"].items():
            layer = int(layer_key)
            layer_type = "final" if layer == n_layers - 1 else "fitted"
            for rank, item in enumerate(layer_items):
                writer.writerow([
                    pos, token, tid, layer, layer_type,
                    rank, item["token"], item["token_id"],
                    f"{item['probability']:.8f}",
                ])
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("model", help="HF model id or local path")
    parser.add_argument("lens", help="path to a fitted JacobianLens (.pt file)")
    parser.add_argument("--prompt", "-p", default=None, help="input text (default: stdin)")
    parser.add_argument(
        "--format", "-f",
        choices=("text", "json", "csv"),
        default="text",
        help="output format (default: text)",
    )
    parser.add_argument("--output", "-o", default=None, help="write to file instead of stdout")
    parser.add_argument("--top_k", "-k", type=int, default=5, help="top predictions per layer (default: 5)")
    parser.add_argument("--max_seq_len", type=int, default=512, help="max tokens to process")
    parser.add_argument(
        "--layers", default=None,
        help="comma-separated layer indices to show (default: all fitted + final)",
    )
    parser.add_argument("--device", default=None, help="device to use (default: auto-detect)")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--no_bos", action="store_true", help="do not prepend BOS token")
    parser.add_argument("--verbose", "-v", action="store_true", help="show logging on stderr")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.verbose:
        jlens.configure_logging()
        logging.getLogger("jlens").setLevel(logging.INFO)

    # Get prompt
    if args.prompt is not None:
        prompt = args.prompt
    elif not sys.stdin.isatty():
        prompt = sys.stdin.read()
    else:
        print("Error: provide --prompt or pipe text via stdin", file=sys.stderr)
        sys.exit(1)

    if not prompt.strip():
        print("Error: empty prompt", file=sys.stderr)
        sys.exit(1)

    # Resolve device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    # Device map handling (mirrors fit.py)
    single_gpu = device == "cuda"
    load_kwargs: dict = {"torch_dtype": dtype}
    if not single_gpu:
        load_kwargs["device_map"] = device

    # Load model
    print(f"Loading model...", file=sys.stderr)
    hf = transformers.AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    if single_gpu:
        hf = hf.cuda()
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    model = jlens.from_hf(hf, tok, force_bos=not args.no_bos)
    n_layers = model.n_layers
    print(f"Model: {model!r}", file=sys.stderr)

    # Load lens
    print(f"Loading lens...", file=sys.stderr)
    lens = jlens.JacobianLens.load(args.lens)
    print(f"Lens: {lens!r}", file=sys.stderr)

    # Parse requested layers
    if args.layers is not None:
        requested_layers = sorted(int(x.strip()) for x in args.layers.split(","))
    else:
        requested_layers = list(lens.source_layers)
        if (n_layers - 1) not in requested_layers:
            requested_layers.append(n_layers - 1)
        requested_layers = sorted(set(requested_layers))

    # Validate layers
    out_of_range = [l for l in requested_layers if not (0 <= l < n_layers)]
    if out_of_range:
        print(f"Error: layers {out_of_range} out of range for {n_layers}-layer model", file=sys.stderr)
        sys.exit(1)

    # Run lens
    top_k = args.top_k
    print(f"Running lens on {len(prompt)} chars...", file=sys.stderr)
    lens_logits, model_logits, input_ids = lens.apply(
        model, prompt,
        layers=[l for l in requested_layers if l in lens.source_layers],
        max_seq_len=args.max_seq_len,
    )
    n_pos = input_ids.shape[1]
    print(f"Sequence: {n_pos} tokens", file=sys.stderr)

    # Ensure final layer is in lens_logits for display
    final_layer = n_layers - 1
    if final_layer in requested_layers and final_layer not in lens_logits:
        lens_logits[final_layer] = model_logits

    # Filter to requested layers only
    lens_logits = {l: v for l, v in lens_logits.items() if l in requested_layers}

    # Extract data
    pos_data = _extract_data(lens_logits, model_logits, input_ids, tok, top_k)

    # Format output
    if args.format == "text":
        output = _format_text(pos_data, requested_layers, top_k, n_layers)
    elif args.format == "json":
        meta = {
            "model": args.model,
            "lens": args.lens,
            "prompt": prompt,
            "n_positions": n_pos,
            "n_layers": n_layers,
            "source_layers": sorted(lens.source_layers),
            "displayed_layers": requested_layers,
            "top_k": top_k,
            "vocab_size": model_logits.shape[-1],
        }
        output = _format_json(pos_data, meta)
    elif args.format == "csv":
        output = _format_csv(pos_data, n_layers, top_k)
    else:
        print(f"Error: unknown format {args.format!r}", file=sys.stderr)
        sys.exit(1)

    # Write output
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
            if not output.endswith("\n"):
                f.write("\n")
        print(f"Output written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
