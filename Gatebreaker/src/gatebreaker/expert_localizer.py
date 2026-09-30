"""
GateBreaker Stage 2: Expert-level Localization.

Localizes safety neurons within previously identified safety experts by
comparing neuron activation patterns between harmful and benign prompts.
(Section 3.3.2, Equations 7-11 of the paper.)
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm

from .config import GateBreakerConfig
from .model_patcher import MoEPatcher


class ExpertLocalizer:
    """
    Stage 2: Expert-level localization to find safety neurons.

    Within identified safety experts, localizes specific neurons
    whose activations discriminate between harmful (refusal) and
    benign (normal) behavior.
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

    # ------------------------------------------------------------------
    # Activation collection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def collect_activations(
        self,
        prompts: List[str],
        safety_experts: Dict[int, List[int]],
    ) -> Dict[Tuple[int, int], List[torch.Tensor]]:
        """Collect per-expert activation signatures for a set of prompts.

        For each prompt:
        1. Run model forward with patcher active
        2. For each safety expert, retrieve activations of routed tokens
        3. Aggregate via element-wise max to produce signature ``v_{l,j}(q)``
           (Eq. 8 for sparse experts, Eq. 9 for shared experts)

        Args:
            prompts: List of prompt strings.
            safety_experts: ``{layer_idx: [expert_indices]}`` from gate profiling.

        Returns:
            ``{(layer, expert): [signature_vector_per_prompt]}``
        """
        self.model.eval()

        # Initialise container
        signatures: Dict[Tuple[int, int], List[torch.Tensor]] = {}
        for layer, experts in safety_experts.items():
            for exp in experts:
                signatures[(layer, exp)] = []

        for prompt in tqdm(prompts, desc="Collecting activations"):
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

            self.patcher.clear_cache()
            self.patcher.patch()
            self.model(**inputs)
            self.patcher.unpatch()

            for layer, experts in safety_experts.items():
                for exp in experts:
                    # Aggregate activations across target sublayers
                    all_acts: List[torch.Tensor] = []
                    for sublayer in self.config.target_sublayers:
                        acts = self.patcher.get_expert_activations(layer, exp, sublayer)
                        if acts is not None and acts.numel() > 0:
                            all_acts.append(acts)

                    if all_acts:
                        # Concatenate across sublayers and aggregate
                        # Each acts tensor: [num_tokens, hidden_dim]
                        combined = torch.cat(all_acts, dim=0)  # [total_tokens, d]
                        # Eq. 8/9: element-wise max aggregation
                        v_lj = combined.max(dim=0).values  # [d]
                    else:
                        # Expert received no tokens for this prompt
                        v_lj = torch.zeros(1)

                    signatures[(layer, exp)].append(v_lj.cpu())

        return signatures

    # ------------------------------------------------------------------
    # Safety weight computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_safety_weights(
        harmful_signatures: Dict[Tuple[int, int], List[torch.Tensor]],
        benign_signatures: Dict[Tuple[int, int], List[torch.Tensor]],
    ) -> Dict[Tuple[int, int], torch.Tensor]:
        """
        Compute per-neuron safety weight for each safety expert (Eq. 10).

        ``w_{l,j,n} = E[v_{l,j}(M)_n] - E[v_{l,j}(B)_n]``

        Returns:
            ``{(layer, expert): safety_weight_tensor}``  shape ``[d_expert]``
        """
        safety_weights: Dict[Tuple[int, int], torch.Tensor] = {}

        for key in harmful_signatures:
            h_sigs = harmful_signatures[key]
            b_sigs = benign_signatures.get(key, [])
            if not h_sigs or not b_sigs:
                continue

            # Filter out degenerate zero-dim placeholder vectors
            h_valid = [s for s in h_sigs if s.numel() > 1]
            b_valid = [s for s in b_sigs if s.numel() > 1]
            if not h_valid or not b_valid:
                continue

            h_tensor = torch.stack(h_valid)  # [N_harm, d]
            b_tensor = torch.stack(b_valid)  # [N_benign, d]

            w_lj = h_tensor.mean(dim=0) - b_tensor.mean(dim=0)  # Eq. 10
            safety_weights[key] = w_lj

        return safety_weights

    # ------------------------------------------------------------------
    # Safety neuron identification
    # ------------------------------------------------------------------

    @staticmethod
    def identify_safety_neurons(
        safety_weights: Dict[Tuple[int, int], torch.Tensor],
        tau: float = 2.0,
    ) -> Tuple[
        Dict[Tuple[int, int], torch.Tensor],
        Dict[Tuple[int, int], torch.Tensor],
    ]:
        """
        Identify safety neurons via z-score thresholding (Eq. 11).

        For each expert, normalise the safety weights to z-scores and
        select neurons with ``z > tau``.

        Returns:
            ``(safety_neurons, z_scores)`` where

            - ``safety_neurons``: ``{(layer, expert): index_tensor}``
            - ``z_scores``: ``{(layer, expert): z_score_tensor}``
        """
        safety_neurons: Dict[Tuple[int, int], torch.Tensor] = {}
        z_scores_dict: Dict[Tuple[int, int], torch.Tensor] = {}

        for key, w_lj in safety_weights.items():
            mu = w_lj.mean()
            sigma = w_lj.std()
            if sigma < 1e-8:
                z = torch.zeros_like(w_lj)
            else:
                z = (w_lj - mu) / sigma  # Eq. 11

            neuron_indices = torch.nonzero(z > tau, as_tuple=False).squeeze(-1)
            safety_neurons[key] = neuron_indices
            z_scores_dict[key] = z

        return safety_neurons, z_scores_dict

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def localize(
        self,
        harmful_prompts: List[str],
        benign_prompts: List[str],
        safety_experts: Dict[int, List[int]],
    ) -> Dict[str, Any]:
        """Run the full localization pipeline.

        Returns:
            ``{'safety_neurons': …, 'safety_weights': …,
              'z_scores': …, 'neuron_ratio': float}``
        """
        tau = self.config.z_threshold

        # 1. Collect activation signatures
        harmful_sigs = self.collect_activations(harmful_prompts, safety_experts)
        benign_sigs = self.collect_activations(benign_prompts, safety_experts)

        # 2. Compute safety weights (Eq. 10)
        safety_weights = self.compute_safety_weights(harmful_sigs, benign_sigs)

        # 3. Identify safety neurons (Eq. 11)
        safety_neurons, z_scores = self.identify_safety_neurons(safety_weights, tau)

        # 4. Compute neuron ratio
        total_neurons = 0
        total_safety = 0
        for key, indices in safety_neurons.items():
            w = safety_weights[key]
            total_neurons += w.shape[0]
            total_safety += len(indices)

        neuron_ratio = total_safety / total_neurons if total_neurons > 0 else 0.0

        return {
            "safety_neurons": safety_neurons,
            "safety_weights": safety_weights,
            "z_scores": z_scores,
            "neuron_ratio": neuron_ratio,
        }
