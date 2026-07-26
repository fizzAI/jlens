# jlens — Jacobian lens (fork)

*This is a fork of the original [jlens](https://github.com/anthropics/jlens) repository that integrates minor changes from Neuronpedia's version of the code, as well as some other minor adjustments.*

Companion code for [**Verbalizable Representations Form a Global Workspace in
Language Models**](https://transformer-circuits.pub/2026/workspace/index.html).

The Jacobian lens maps an internal activation vector (at any layer and position)
into the final-layer vocabulary space, then decodes it with the unembedding
matrix into a ranked list of tokens. This shows you what a given activation is
poised to contribute to the model's output.

The mapping is the average input–output Jacobian over a text corpus:

$`\text{lens}_l(h) = \text{unembed}(J_l \cdot h), \quad J_l = \mathbb{E}\left[\frac{\partial h_{\text{final}}}{\partial h_l}\right]`$

The expectation is over prompts, source positions, and all current-and-future
target positions in a generic web-text corpus; the estimator used
(cotangents summed over target positions, then averaged over source positions)
is documented in [`jlens.fitting`](jlens/fitting.py).

This repo allows you to use and fit lens on open-weights decoder transformers, via
the `transformers` library.

## Install

```bash
uv sync # or uv pip install -e .
```

For fitting, it's recommended to install with the `fit` extra:

```bash
uv sync --extra fit # or uv pip install -e .[fit]
```

## Usage as a Library

### Infer

To run inference with a pre-fitted lens:

```python
import transformers, jlens

hf = transformers.AutoModelForCausalLM.from_pretrained("org/model").cuda()
tok = transformers.AutoTokenizer.from_pretrained("org/model")
model = jlens.from_hf(hf, tok)

lens = jlens.JacobianLens.from_pretrained("org/lens-repo", filename="model/lens.pt")
lens_logits, model_logits, _ = lens.apply(
    model, "Fact: The currency used in the country shaped like a boot is",
    positions=[-2])
for layer, logits in sorted(lens_logits.items()):
    print(layer, [tok.decode([t]) for t in logits[0].topk(5).indices])
```

Image-text-to-text models are also supported, however they strip off the multimodal encoders and only allow you to use text functionality via the library.

```python
import transformers, jlens

hf = transformers.AutoModelForImageTextToText.from_pretrained("org/model").cuda()
tok = transformers.AutoProcessor.from_pretrained("org/model")
model = jlens.from_hf(hf, tok)

lens = jlens.JacobianLens.from_pretrained("org/lens-repo", filename="model/lens.pt")
lens_logits, model_logits, _ = lens.apply(
    model, "Did you know that the Eiffel Tower is located in the city of",
    positions=[-2])
for layer, logits in sorted(lens_logits.items()):
    print(layer, [tok.decode([t]) for t in logits[0].topk(5).indices])
```

### Fit

To fit a lens on your own model:

```python
my_prompts = [
    "Blah blah blah blah..."
]

lens = jlens.fit(model, prompts=my_prompts, checkpoint_path="out/ckpt.pt")
lens.save("out/jacobian_lens.pt")
```

The paper's lenses use 1000 sequences of 128 tokens from a pretraining-like
corpus. Quality saturates quickly (see section 9.3 of the paper), though, and
~100 prompts is usable. You can parallelize fitting by running multiple `fit()`s
on different slices of your data, then combining all of them with `JacobianLens.merge()`.

## Scripts

### `fit.py`

`scripts/fit.py` wraps `jlens.fit` into a CLI that handles model and dataset loading, metrics reporting, and early stopping.

```bash
uv run -m fit Qwen/Qwen3.5-0.8B --out_dir out/
uv run -m fit scripts/fit.py meta-llama/Llama-3.1-8B --n_prompts 1000 --stop_at_delta 1e-3

# actual usage
uv run -m fit XiaomiMiMo/MiMo-7B-RL-0530 --out_dir lenses/mimo --dataset Salesforce/wikitext --dataset_config wikitext-103-raw-v1 --dataset_split train --text_field text --max_chars 2000 --n_prompts 1000 --dim_batch 64 --max_seq_len 128 --dtype bfloat16 --device_map cuda --min_prompts 100 --stop_window 10 --levels 1e-2,5e-3,1e-3 --stop_at_delta 0.002 --trust_remote_code
```

Refer to the help page for information about the command line arguments.

The script writes a metrics log CSV to `<out_dir>/` alongside the fitted lens. The lens is checkpointed after every prompt fit.

## License

Code is released under the Apache License 2.0 (see [LICENSE](LICENSE)).
