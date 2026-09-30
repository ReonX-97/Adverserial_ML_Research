"""
GateBreaker Stage 3: Targeted Safety Removal.

Implements inference-time activation clamping for identified safety neurons
within MoE expert sublayers (Eq. 12, Section 3.3.3 of the paper).

This module is provided strictly for academic reproducibility of the
GateBreaker paper (arXiv:2512.21008v2). It targets only openly released
research models evaluated in the original study.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer

from gatebreaker.config import GateBreakerConfig


class SafetyRemover:
    """Stage 3: Targeted safety removal via inference-time activation clamping.

    Installs lightweight forward hooks on expert sublayers that zero out
    the activation dimensions corresponding to identified safety neurons.
    This does **not** modify model weights; it only masks intermediate
    activations during the forward pass (Eq. 12).

    Attributes:
        model: The target MoE model.
        config: GateBreaker configuration.
        hooks: List of registered hook handles.
        safety_neuron_masks: Mapping from ``(layer, expert, sublayer)`` to
            a boolean tensor where ``True`` marks neurons to be zeroed.
    """

    def __init__(self, model: PreTrainedModel, config: GateBreakerConfig) -> None:
        self.model = model
        self.config = config
        self.hooks: List[torch.utils.hooks.RemovableHook] = []
        self.safety_neuron_masks: Dict[Tuple[int, int, str], torch.Tensor] = {}
        self._applied = False

    # ------------------------------------------------------------------
    # Loading safety neurons
    # ------------------------------------------------------------------

    def load_safety_neurons(
        self,
        safety_neurons: Dict[Tuple[int, int], torch.Tensor],
        sublayers: Optional[List[str]] = None,
    ) -> None:
        """Convert neuron indices into boolean masks for each expert sublayer.

        Args:
            safety_neurons: ``{(layer, expert): index_tensor}`` produced by
                :class:`ExpertLocalizer`.  Each tensor contains the integer
                indices of neurons classified as safety-critical.
            sublayers: Which sublayers to target (default: config.target_sublayers).
        """
        sublayers = sublayers or self.config.target_sublayers
        self.safety_neuron_masks.clear()

        for (layer_idx, expert_idx), neuron_indices in safety_neurons.items():
            # Determine dimensionality from the model
            expert_module = self._get_expert_module(layer_idx, expert_idx)
            if expert_module is None:
                continue

            for sublayer_name in sublayers:
                sub = getattr(expert_module, sublayer_name, None)
                if sub is None:
                    continue
                # The output dim of gate_proj / up_proj = intermediate_size
                out_features = sub.out_features
                mask = torch.zeros(out_features, dtype=torch.bool)
                idx = neuron_indices.long()
                idx = idx[idx < out_features]  # safety clamp
                mask[idx] = True
                key = (layer_idx, expert_idx, sublayer_name)
                self.safety_neuron_masks[key] = mask

    # ------------------------------------------------------------------
    # Apply / remove hooks
    # ------------------------------------------------------------------

    def apply(self) -> None:
        """Install forward hooks that zero out safety neurons at inference time.

        For every ``(layer, expert, sublayer)`` entry in the mask dictionary,
        a hook is registered on the corresponding ``nn.Linear`` module.  The
        hook sets the masked output dimensions to zero *after* the linear
        projection (i.e., it modifies the hook output, not the weights).
        """
        if self._applied:
            return

        device = next(self.model.parameters()).device

        for (layer_idx, expert_idx, sublayer_name), mask in self.safety_neuron_masks.items():
            module = self._get_sublayer_module(layer_idx, expert_idx, sublayer_name)
            if module is None:
                continue

            # Move mask to the correct device
            mask_dev = mask.to(device)

            def _make_hook(m: torch.Tensor):
                """Create a closure that captures the correct mask."""
                def hook_fn(mod: nn.Module, inp: Any, out: torch.Tensor) -> torch.Tensor:
                    # Eq. 12: A'_{l,j}(x)_n = 0 if n in N_safety
                    out = out.clone()
                    out[..., m] = 0.0
                    return out
                return hook_fn

            handle = module.register_forward_hook(_make_hook(mask_dev))
            self.hooks.append(handle)

        self._applied = True

    def remove(self) -> None:
        """Remove all safety-removal hooks, restoring original behaviour."""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()
        self._applied = False

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        tokenizer: PreTrainedTokenizer,
        prompt: str,
        max_new_tokens: int | None = None,
    ) -> str:
        """Generate text with safety neurons disabled.

        The hooks are applied before generation and remain active.

        Args:
            tokenizer: The model's tokenizer.
            prompt: Raw user prompt string.
            max_new_tokens: Override for ``config.max_new_tokens``.

        Returns:
            The generated text (assistant response only).
        """
        if not self._applied:
            self.apply()

        max_new_tokens = max_new_tokens or self.config.max_new_tokens

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )
        response_ids = output_ids[0, input_len:]
        return tokenizer.decode(response_ids, skip_special_tokens=True)

    @torch.no_grad()
    def generate_batch(
        self,
        tokenizer: PreTrainedTokenizer,
        prompts: List[str],
        max_new_tokens: int | None = None,
    ) -> List[str]:
        """Generate responses for multiple prompts with safety neurons disabled."""
        return [
            self.generate(tokenizer, p, max_new_tokens=max_new_tokens)
            for p in prompts
        ]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_neuron_stats(self) -> Dict[str, Any]:
        """Return statistics about the safety neuron intervention.

        Returns:
            Dictionary with total/modified neuron counts, ratio, and per-expert
            breakdown.
        """
        total_modified = 0
        total_in_layers = 0
        per_expert: Dict[Tuple[int, int, str], int] = {}

        for key, mask in self.safety_neuron_masks.items():
            count = int(mask.sum().item())
            total_modified += count
            total_in_layers += mask.numel()
            per_expert[key] = count

        ratio = total_modified / total_in_layers if total_in_layers > 0 else 0.0

        return {
            "total_neurons_modified": total_modified,
            "total_neurons_in_targeted_layers": total_in_layers,
            "safety_neuron_ratio": ratio,
            "per_expert_counts": per_expert,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_expert_module(
        self, layer_idx: int, expert_idx: int
    ) -> nn.Module | None:
        """Resolve the ``nn.Module`` for a given expert."""
        try:
            moe_block = self.model.model.layers[layer_idx].mlp
            # Check if it's a shared expert (negative index convention or
            # explicit attribute).
            if hasattr(moe_block, "shared_expert") and expert_idx == -1:
                return moe_block.shared_expert
            return moe_block.experts[expert_idx]
        except (IndexError, AttributeError):
            return None

    def _get_sublayer_module(
        self, layer_idx: int, expert_idx: int, sublayer_name: str
    ) -> nn.Module | None:
        """Resolve the specific sublayer ``nn.Linear`` of an expert."""
        expert = self._get_expert_module(layer_idx, expert_idx)
        if expert is None:
            return None
        return getattr(expert, sublayer_name, None)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save_masks(self, path: str) -> None:
        """Persist safety neuron masks to disk."""
        torch.save(
            {str(k): v for k, v in self.safety_neuron_masks.items()}, path
        )

    def load_masks(self, path: str) -> None:
        """Load previously saved safety neuron masks."""
        data = torch.load(path, weights_only=True)
        self.safety_neuron_masks = {eval(k): v for k, v in data.items()}
