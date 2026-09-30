"""
GateBreaker Configuration Module.

Defines the central configuration dataclass for the GateBreaker pipeline,
including model selection, hyperparameters, and runtime settings.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class GateBreakerConfig:
    """Configuration for the GateBreaker pipeline.

    Attributes:
        model_name: HuggingFace model identifier for the target MoE LLM.
        num_profiling_prompts: Number of harmful prompts used for gate profiling.
        safety_expert_top_k_multiplier: Multiplier for selecting candidate safety
            experts. For each MoE layer, experts whose average activation frequency
            ranks within the top-(multiplier * k) are selected, where k is the
            model's native top-k routing parameter.
        z_threshold: Tau threshold for z-score based safety neuron identification.
            Neurons with z-score above this threshold are classified as safety neurons.
        aggregation: Aggregation function for combining per-token activations into
            a per-prompt signature. 'max' = element-wise maximum (Eq. 8-9 in paper).
        target_sublayers: Which expert sublayers to analyze for safety neurons.
            The paper focuses on gate_proj and up_proj (Section 7.2).
        batch_size: Batch size for inference during profiling and localization.
        max_new_tokens: Maximum number of new tokens to generate during evaluation.
        device: PyTorch device string.
        dtype: Model dtype string ('bfloat16', 'float16', 'float32').
        output_dir: Directory for saving results, masks, and reports.
        model_top_k: The model's native top-k expert routing parameter.
            Auto-detected from model config if set to None.
        num_sparse_experts: Number of sparse experts per MoE layer.
            Auto-detected from model config if set to None.
        num_shared_experts: Number of shared experts per MoE layer.
            Auto-detected from model config if set to None.
    """

    # Model
    model_name: str = "Qwen/Qwen1.5-MoE-A2.7B-Chat"

    # Dataset
    num_profiling_prompts: int = 200

    # Stage 1: Gate-level profiling
    safety_expert_top_k_multiplier: int = 3

    # Stage 2: Expert-level localization
    z_threshold: float = 2.0
    aggregation: str = "max"
    target_sublayers: List[str] = field(
        default_factory=lambda: ["gate_proj", "up_proj"]
    )

    # Runtime
    batch_size: int = 4
    max_new_tokens: int = 256
    device: str = "cuda"
    dtype: str = "bfloat16"

    # Output
    output_dir: str = "./results"

    # Model architecture (auto-detected if None)
    model_top_k: int | None = None
    num_sparse_experts: int | None = None
    num_shared_experts: int | None = None

    def update_from_model_config(self, model_config) -> None:
        """Auto-populate architecture fields from the HuggingFace model config.

        Args:
            model_config: The model's ``config`` object (e.g.,
                ``Qwen2MoeConfig``).
        """
        if self.model_top_k is None:
            self.model_top_k = getattr(
                model_config, "num_experts_per_tok", 4
            )
        if self.num_sparse_experts is None:
            self.num_sparse_experts = getattr(
                model_config, "num_experts", 60
            )
        if self.num_shared_experts is None:
            self.num_shared_experts = getattr(
                model_config, "num_shared_experts", 4  # Qwen1.5-MoE default
            )

    @property
    def num_safety_experts_per_layer(self) -> int:
        """Number of candidate safety experts selected per MoE layer."""
        k = self.model_top_k or 4
        return self.safety_expert_top_k_multiplier * k

    def __post_init__(self) -> None:
        valid_aggs = {"max", "mean", "sum"}
        if self.aggregation not in valid_aggs:
            raise ValueError(
                f"aggregation must be one of {valid_aggs}, got '{self.aggregation}'"
            )
        valid_sublayers = {"gate_proj", "up_proj", "down_proj"}
        for sl in self.target_sublayers:
            if sl not in valid_sublayers:
                raise ValueError(
                    f"Invalid sublayer '{sl}'. Must be one of {valid_sublayers}"
                )
