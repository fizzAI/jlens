# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Fitting tests against real (small, randomly-initialised) HuggingFace models.

Each entry in ``_MODELS`` is a ``(name, repo_id, source_layers)`` triple.
To test against a new model, add another tuple to that list. The model is
loaded once per module via the ``hf_model`` fixture.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModelForImageTextToText

from jlens import HFLensModel, JacobianLens, fit, from_hf

_MODELS: list[tuple[str, str, list[int], str]] = [
    ("llama-3.3-tiny-random", "yujiepan/llama-3.3-tiny-random", [0], "causal_lm"),
    # ("deepseek-v4-tiny-random", "yujiepan/deepseek-v4-bf16-tiny-random", [0,1,2,3,4,5]), # doesn't work? not for jlens reasons just doesn't work on my machine
    ("qwen3.6-tiny-random", "yujiepan/qwen3.6-tiny-random", [0, 1, 2], "conditional_generation"),
    ("glm-5.2-tiny-random", "yujiepan/glm-5.2-tiny-random", [0, 1, 2], "causal_lm"),
]


def _model_ids() -> list[str]:
    return [name for name, _, _, _ in _MODELS]


_PROMPTS: list[str] = [
    "The quick brown fox jumps over the lazy dog near the river bank on a sunny afternoon in the countryside.",
    "A second sentence for testing convergence behavior with the Jacobian lens fitting procedure across multiple prompts.",
    "Yet another prompt to accumulate more data points for our running mean Jacobian estimation to converge properly.",
    "In machine learning we often need to verify that iterative averaging procedures converge to stable values over time.",
    "The transformer architecture processes sequences of tokens through multiple layers of self-attention and feed-forward networks.",
]


@pytest.fixture(scope="module", params=_MODELS, ids=_model_ids())
def hf_model(request) -> tuple[HFLensModel, list[int]]:
    """Load each registered model once for the whole module.

    Returns ``(model, source_layers)`` for the current registry entry.
    """
    _, repo_id, source_layers, model_type = request.param # pyright: ignore
    auto_cls = None
    tok_cls = None
    match model_type: # pyright: ignore
        case "causal_lm":
            auto_cls = AutoModelForCausalLM
            tok_cls = AutoTokenizer
        case "conditional_generation":
            auto_cls = AutoModelForImageTextToText # pyright: ignore
            tok_cls = AutoProcessor
    model = auto_cls.from_pretrained(repo_id) # pyright: ignore
    tok = tok_cls.from_pretrained(repo_id) # pyright: ignore
    return from_hf(model, tok), source_layers # pyright: ignore


def _collect_mean_rel_changes(
    model, prompts: list[str], **fit_kwargs: Any
) -> list[float]:
    """Run ``fit()`` and return all ``mean_rel_change`` values (including NaN)."""
    changes: list[float] = []

    def _cb(progress, _extra):
        changes.append(progress.mean_rel_change)

    fit(model, prompts, metrics_callback=_cb, **fit_kwargs)
    return changes


class TestConvergence:
    """The running-mean Jacobian stabilises as more prompts are accumulated."""

    def test_mean_rel_change_decreases(self, hf_model):
        """Ending ``mean_rel_change`` must be lower than the first finite one."""
        model, layers = hf_model
        changes = _collect_mean_rel_changes(
            model, _PROMPTS, source_layers=layers, dim_batch=4, max_seq_len=64
        )
        finite = [c for c in changes if c == c]  # drop NaN (first prompt)
        assert len(finite) >= 2, (
            f"need at least 2 finite changes, got {len(finite)}"
        )
        assert finite[-1] < finite[0], (
            f"ending mean_rel_change ({finite[-1]}) not lower than "
            f"starting ({finite[0]})"
        )

    def test_first_change_is_reasonably_large(self, hf_model):
        """The first finite change should be meaningfully above zero."""
        model, layers = hf_model
        changes = _collect_mean_rel_changes(
            model, _PROMPTS, source_layers=layers, dim_batch=4, max_seq_len=64
        )
        finite = [c for c in changes if c == c]
        assert finite[0] > 1e-4, (
            f"first change suspiciously small: {finite[0]}"
        )


class TestFitOutput:
    """fit() produces a valid JacobianLens with expected metadata."""

    def test_lens_metadata(self, hf_model):
        model, layers = hf_model
        lens = fit(
            model, _PROMPTS, source_layers=layers, dim_batch=4, max_seq_len=64
        )
        assert isinstance(lens, JacobianLens)
        assert lens.source_layers == layers
        assert lens.n_prompts == len(_PROMPTS)
        assert lens.d_model == model.d_model

    def test_jacobian_shape_and_dtype(self, hf_model):
        model, layers = hf_model
        lens = fit(
            model, _PROMPTS, source_layers=layers, dim_batch=4, max_seq_len=64
        )
        J = lens.jacobians[layers[0]]
        assert J.shape == (model.d_model, model.d_model)
        assert J.dtype == torch.float32

    def test_late_layer_close_to_identity(self, hf_model):
        """For a shallow randomly-initialised model the last fitted layer's
        Jacobian should stay close to the identity (residual connection
        dominates)."""
        model, layers = hf_model
        lens = fit(
            model, _PROMPTS, source_layers=layers, dim_batch=4, max_seq_len=64
        )
        J = lens.jacobians[layers[-1]]
        identity_dist = (J - torch.eye(model.d_model)).norm().item()
        # Generous bound: the small random perturbations accumulate but the
        # residual structure keeps it near I.
        assert identity_dist < 2.0, (
            f"J_{layers[-1]} too far from identity: {identity_dist}"
        )


class TestApply:
    """A fitted lens can be applied to produce logits of the right shape."""

    def test_apply_returns_correct_shapes(self, hf_model):
        model, layers = hf_model
        lens = fit(
            model, _PROMPTS, source_layers=layers, dim_batch=4, max_seq_len=64
        )
        test_prompt = "hello world testing the fitted lens application"
        lens_logits, model_logits, input_ids = lens.apply(
            model, test_prompt, layers=layers, max_seq_len=64
        )
        seq_len = input_ids.shape[1]
        vocab_size = model._lm_head.out_features
        assert model_logits.shape == (seq_len, vocab_size)
        assert set(lens_logits) == set(layers)
        assert lens_logits[layers[0]].shape == (seq_len, vocab_size)

    def test_apply_with_positions_subset(self, hf_model):
        model, layers = hf_model
        lens = fit(
            model, _PROMPTS, source_layers=layers, dim_batch=4, max_seq_len=64
        )
        test_prompt = "hello world testing the fitted lens application"
        lens_logits, model_logits, _ = lens.apply(
            model, test_prompt, layers=layers, positions=[0, -1], max_seq_len=64
        )
        assert model_logits.shape[0] == 2
        assert lens_logits[layers[0]].shape[0] == 2


class TestCheckpointResume:
    """fit() can save and resume from a checkpoint identically."""

    def test_resume_produces_same_result(self, hf_model, tmp_path):
        model, layers = hf_model
        ckpt = str(tmp_path / "ckpt.pt")
        prompts_subset = _PROMPTS[:2]

        # First run: fit two prompts, saving checkpoint.
        fit(
            model, prompts_subset, source_layers=layers, dim_batch=4,
            max_seq_len=64, checkpoint_path=ckpt,
        )

        # Second run: resume with all five prompts. Checkpoint covers the
        # first two, so only the remaining three are processed.
        resumed = fit(
            model, _PROMPTS, source_layers=layers, dim_batch=4,
            max_seq_len=64, checkpoint_path=ckpt,
        )

        # Reference: fit all five from scratch.
        reference = fit(
            model, _PROMPTS, source_layers=layers, dim_batch=4, max_seq_len=64,
        )

        assert resumed.n_prompts == reference.n_prompts == len(_PROMPTS)
        torch.testing.assert_close(
            resumed.jacobians[layers[0]], reference.jacobians[layers[0]],
            rtol=0, atol=1e-5,
        )