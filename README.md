# jlens — Jacobian lens (fork)

> This is a fork of the original [jlens](https://github.com/anthropics/jlens) repository. This code is not maintained and is not accepting contributions.

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
pip install -e .
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

### Fit

To fit a lens on your own model:

```python
lens = jlens.fit(model, prompts=my_prompts, checkpoint_path="out/ckpt.pt")
lens.save("out/jacobian_lens.pt")
```

The paper's lenses use 1000 sequences of 128 tokens from a pretraining-like
corpus. Quality saturates quickly (see section 9.3 of the paper); ~100 prompts
is usable. This is a reference implementation and is not optimized; fitting
time is dominated by the model's own backward pass. Parallelize by running
`fit()` on disjoint slices and combining with `JacobianLens.merge()`.

## License

Code is released under the Apache License 2.0 (see [LICENSE](LICENSE)).
