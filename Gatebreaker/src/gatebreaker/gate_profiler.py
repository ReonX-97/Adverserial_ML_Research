"""
GateBreaker Stage 1: Gate-level Profiling.

Implements the first stage of the GateBreaker pipeline which identifies
safety experts by analyzing gate activation patterns under harmful prompts.
(Section 3.3.1, Equations 4-6 of the paper.)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm

from .config import GateBreakerConfig
from .datasets import mask_non_content_tokens
from .model_patcher import MoEPatcher


class GateProfiler:
    """
    Stage 1: Gate-level profiling to identify safety experts.

    Analyses gate activation patterns under harmful prompts to find
    experts that are disproportionately selected when the model
    encounters harmful content.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        patcher: MoEPatcher,
        config: GateBreakerConfig,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.patcher = patcher
        self.config = config
        self.results: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Main profiling entry-point
    # ------------------------------------------------------------------

    @torch.no_grad()
    def profile(self, harmful_prompts: List[str]) -> Dict[str, Any]:
        """
        Run gate-level profiling on harmful prompts.

        For each prompt the model is run in inference mode with the patcher
        active.  Routing decisions are recorded per token, then converted
        into activation counts (Eq. 4), frequencies (Eq. 5), and expert
        utility scores (Eq. 6).

        Returns:
            ``{'activation_counts': …, 'activation_frequencies': …,
              'expert_utility_scores': …}``
        """
        self.model.eval()
        per_prompt_frequencies: Dict[Tuple[int, int], List[float]] = defaultdict(list)

        for prompt in tqdm(harmful_prompts, desc="Gate profiling"):
            # Build chat-formatted input
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            seq_len = inputs["input_ids"].shape[1]

            # Content-token mask (Section 4.2)
            content_mask = mask_non_content_tokens(self.tokenizer, prompt)
            # Pad / truncate mask to match actual sequence length
            if len(content_mask) < seq_len:
                pad = torch.zeros(seq_len - len(content_mask), dtype=torch.bool)
                content_mask = torch.cat([content_mask, pad])
            else:
                content_mask = content_mask[:seq_len]

            L = int(content_mask.sum().item())
            if L == 0:
                continue

            # Forward pass with hooks active
            self.patcher.clear_cache()
            self.patcher.patch()
            self.model(**inputs)
            self.patcher.unpatch()

            # Process routing decisions for each layer
            routing = self.patcher.get_routing_decisions()
            for layer_idx, expert_indices in routing.items():
                # expert_indices: [num_tokens, top_k]
                # Select only content tokens
                if expert_indices.shape[0] >= seq_len:
                    layer_routing = expert_indices[:seq_len][content_mask]
                else:
                    # If flattened across batch, take what we can
                    layer_routing = expert_indices[:seq_len]

                # Count how often each expert was selected (Eq. 4)
                flat = layer_routing.flatten()
                for expert_id in flat.unique().tolist():
                    count = int((flat == expert_id).sum().item())
                    f_lj = count / L  # Eq. 5
                    per_prompt_frequencies[(layer_idx, expert_id)].append(f_lj)

        # Compute expert utility scores (Eq. 6)
        num_prompts = len(harmful_prompts)
        expert_utility_scores: Dict[Tuple[int, int], float] = {}
        activation_frequencies: Dict[Tuple[int, int], float] = {}

        for key, freq_list in per_prompt_frequencies.items():
            u_lj = sum(freq_list) / num_prompts
            expert_utility_scores[key] = u_lj
            activation_frequencies[key] = sum(freq_list) / len(freq_list)

        self.results = {
            "activation_frequencies": activation_frequencies,
            "expert_utility_scores": expert_utility_scores,
        }
        return self.results

    # ------------------------------------------------------------------
    # Safety expert selection
    # ------------------------------------------------------------------

    def select_safety_experts(
        self,
        utility_scores: Dict[Tuple[int, int], float] | None = None,
        top_k_multiplier: int | None = None,
    ) -> Dict[int, List[int]]:
        """
        Select candidate safety experts per layer.

        For each MoE layer, experts whose average activation frequency
        ranks within the top-``multiplier * k`` are selected (where *k*
        is the model's native top-k routing parameter).

        Returns:
            ``{layer_idx: [expert_indices]}``
        """
        utility_scores = utility_scores or self.results.get("expert_utility_scores", {})
        top_k_multiplier = top_k_multiplier or self.config.safety_expert_top_k_multiplier

        # Group by layer
        layer_experts: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        for (layer, exp), score in utility_scores.items():
            layer_experts[layer].append((exp, score))

        k = self.config.model_top_k or self.patcher.top_k
        limit = k * top_k_multiplier

        selected: Dict[int, List[int]] = {}
        for layer, exp_scores in layer_experts.items():
            exp_scores.sort(key=lambda x: x[1], reverse=True)
            selected[layer] = [exp for exp, _ in exp_scores[:limit]]

        self.results["selected_experts"] = selected
        return selected

    def get_profiling_results(self) -> Dict[str, Any]:
        """Return all profiling results."""
        return self.results
