"""
GateBreaker — Modal Cloud GPU Deployment.

Runs the full GateBreaker pipeline (gate profiling → expert localization →
safety removal → evaluation) on a Modal cloud GPU.

Usage:
    # Set up your Modal account first: `modal setup`
    # Then run:
    modal run modal_app.py
"""

from __future__ import annotations

import json
import os
import sys

import modal

# ---------------------------------------------------------------------------
# Modal configuration
# ---------------------------------------------------------------------------

# Docker image with all dependencies pre-installed
gatebreaker_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "accelerate>=0.27.0",
        "datasets>=2.16.0",
        "tqdm>=4.65.0",
    )
    .copy_local_dir("src", "/root/src")
)

app = modal.App("gatebreaker", image=gatebreaker_image)

# Volume for caching HuggingFace models
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

# ---------------------------------------------------------------------------
# Helper — load model & tokenizer
# ---------------------------------------------------------------------------


def _load_model(model_name: str, device: str = "cuda"):
    """Load the target MoE model and tokenizer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir="/cache/huggingface",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        cache_dir="/cache/huggingface",
    )
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Stage 1 — Gate-level profiling
# ---------------------------------------------------------------------------


@app.function(
    gpu="A10G",
    timeout=3600,
    volumes={"/cache": hf_cache},
)
def run_gate_profiling(
    model_name: str = "Qwen/Qwen1.5-MoE-A2.7B-Chat",
    num_prompts: int = 100,
):
    """Run Stage 1: gate-level profiling on cloud GPU."""
    sys.path.insert(0, "/root/src")

    from gatebreaker.config import GateBreakerConfig
    from gatebreaker.datasets import load_harmful_prompts
    from gatebreaker.gate_profiler import GateProfiler
    from gatebreaker.model_patcher import MoEPatcher

    print(f"[Stage 1] Loading model {model_name} ...")
    model, tokenizer = _load_model(model_name)

    config = GateBreakerConfig(model_name=model_name, num_profiling_prompts=num_prompts)
    config.update_from_model_config(model.config)

    patcher = MoEPatcher(model, model.config)
    profiler = GateProfiler(model, tokenizer, patcher, config)

    harmful_prompts = load_harmful_prompts(num_prompts)
    print(f"[Stage 1] Profiling {len(harmful_prompts)} harmful prompts ...")

    results = profiler.profile(harmful_prompts)
    safety_experts = profiler.select_safety_experts()

    print(f"[Stage 1] Found safety experts in {len(safety_experts)} layers")
    for layer, experts in sorted(safety_experts.items()):
        print(f"  Layer {layer}: experts {experts}")

    # Serialise results for cross-function transfer
    serialisable = {
        "expert_utility_scores": {
            f"{k[0]}_{k[1]}": v
            for k, v in results["expert_utility_scores"].items()
        },
        "safety_experts": {str(k): v for k, v in safety_experts.items()},
    }
    return serialisable


# ---------------------------------------------------------------------------
# Stage 2+3 — Localization & Safety removal + Evaluation
# ---------------------------------------------------------------------------


@app.function(
    gpu="A10G",
    timeout=7200,
    volumes={"/cache": hf_cache},
)
def run_localization_and_attack(
    model_name: str = "Qwen/Qwen1.5-MoE-A2.7B-Chat",
    profiling_results: dict | None = None,
    num_prompts: int = 100,
    z_threshold: float = 2.0,
):
    """Run Stages 2 & 3: expert localization → safety removal → evaluation."""
    sys.path.insert(0, "/root/src")

    import torch
    from gatebreaker.config import GateBreakerConfig
    from gatebreaker.datasets import load_harmful_prompts, load_benign_prompts
    from gatebreaker.expert_localizer import ExpertLocalizer
    from gatebreaker.metrics import compute_asr, GateBreakerEvaluator
    from gatebreaker.model_patcher import MoEPatcher
    from gatebreaker.safety_remover import SafetyRemover

    print(f"[Stage 2-3] Loading model {model_name} ...")
    model, tokenizer = _load_model(model_name)

    config = GateBreakerConfig(
        model_name=model_name,
        z_threshold=z_threshold,
        num_profiling_prompts=num_prompts,
    )
    config.update_from_model_config(model.config)

    patcher = MoEPatcher(model, model.config)

    # Reconstruct safety_experts dict
    if profiling_results and "safety_experts" in profiling_results:
        safety_experts = {
            int(k): v for k, v in profiling_results["safety_experts"].items()
        }
    else:
        # Re-run profiling if results not provided
        from gatebreaker.gate_profiler import GateProfiler

        profiler = GateProfiler(model, tokenizer, patcher, config)
        harmful = load_harmful_prompts(num_prompts)
        profiler.profile(harmful)
        safety_experts = profiler.select_safety_experts()

    # --- Stage 2: Expert-level localization ---
    print("[Stage 2] Localizing safety neurons ...")
    localizer = ExpertLocalizer(model, tokenizer, patcher, config)
    harmful_prompts = load_harmful_prompts(num_prompts)
    benign_prompts = load_benign_prompts(num_prompts)

    loc_results = localizer.localize(harmful_prompts, benign_prompts, safety_experts)
    safety_neurons = loc_results["safety_neurons"]

    total_neurons = sum(len(v) for v in safety_neurons.values())
    print(f"[Stage 2] Identified {total_neurons} safety neurons "
          f"(ratio: {loc_results['neuron_ratio']:.4f})")

    # --- Baseline responses (before attack) ---
    print("[Eval] Generating baseline responses ...")
    eval_prompts = harmful_prompts[:50]  # evaluate on subset
    baseline_responses = []
    for p in eval_prompts:
        messages = [{"role": "user", "content": p}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        resp = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        baseline_responses.append(resp)

    baseline_asr = compute_asr(baseline_responses)
    print(f"[Eval] Baseline ASR: {baseline_asr:.2%}")

    # --- Stage 3: Targeted safety removal ---
    print("[Stage 3] Applying targeted safety removal ...")
    remover = SafetyRemover(model, config)
    remover.load_safety_neurons(safety_neurons)
    remover.apply()

    stats = remover.get_neuron_stats()
    print(f"[Stage 3] Modified {stats['total_neurons_modified']} neurons "
          f"({stats['safety_neuron_ratio']:.2%} of targeted layers)")

    # --- Post-attack responses ---
    print("[Eval] Generating post-attack responses ...")
    attack_responses = remover.generate_batch(tokenizer, eval_prompts)
    attack_asr = compute_asr(attack_responses)
    print(f"[Eval] Post-attack ASR: {attack_asr:.2%}")

    # --- Full report ---
    evaluator = GateBreakerEvaluator(config)
    report = evaluator.full_report(
        eval_prompts, baseline_responses, attack_responses, safety_neurons, model
    )

    print("\n" + "=" * 60)
    print("GateBreaker Results")
    print("=" * 60)
    print(f"  Model:              {model_name}")
    print(f"  Baseline ASR:       {report['baseline_asr']:.2%}")
    print(f"  Post-attack ASR:    {report['attack_asr']:.2%}")
    print(f"  ASR Improvement:    {report['asr_improvement']:.2%}")
    print(f"  Safety Neuron Ratio:{report['safety_neuron_ratio']:.2%}")
    print(f"  Total Prompts:      {report['total_prompts']}")
    print("=" * 60)

    # Clean up hooks
    remover.remove()

    return {
        "report": report,
        "sample_baseline": baseline_responses[:5],
        "sample_attack": attack_responses[:5],
    }


# ---------------------------------------------------------------------------
# Full pipeline orchestrator
# ---------------------------------------------------------------------------


@app.function(
    gpu="A10G",
    timeout=10800,
    volumes={"/cache": hf_cache},
)
def run_full_pipeline(
    model_name: str = "Qwen/Qwen1.5-MoE-A2.7B-Chat",
    num_prompts: int = 100,
    z_threshold: float = 2.0,
):
    """Run the complete GateBreaker pipeline end-to-end on one GPU."""
    sys.path.insert(0, "/root/src")

    import torch
    from gatebreaker.config import GateBreakerConfig
    from gatebreaker.datasets import load_harmful_prompts, load_benign_prompts
    from gatebreaker.expert_localizer import ExpertLocalizer
    from gatebreaker.gate_profiler import GateProfiler
    from gatebreaker.metrics import compute_asr, GateBreakerEvaluator
    from gatebreaker.model_patcher import MoEPatcher
    from gatebreaker.safety_remover import SafetyRemover

    # Load model
    print(f"Loading model: {model_name}")
    model, tokenizer = _load_model(model_name)

    config = GateBreakerConfig(
        model_name=model_name,
        z_threshold=z_threshold,
        num_profiling_prompts=num_prompts,
    )
    config.update_from_model_config(model.config)
    patcher = MoEPatcher(model, model.config)

    # Load datasets
    harmful_prompts = load_harmful_prompts(num_prompts)
    benign_prompts = load_benign_prompts(num_prompts)

    # Stage 1: Gate profiling
    print("\n--- Stage 1: Gate-level Profiling ---")
    profiler = GateProfiler(model, tokenizer, patcher, config)
    profiler.profile(harmful_prompts)
    safety_experts = profiler.select_safety_experts()
    print(f"Selected safety experts in {len(safety_experts)} layers")

    # Stage 2: Expert localization
    print("\n--- Stage 2: Expert-level Localization ---")
    localizer = ExpertLocalizer(model, tokenizer, patcher, config)
    loc_results = localizer.localize(harmful_prompts, benign_prompts, safety_experts)
    safety_neurons = loc_results["safety_neurons"]
    print(f"Found {sum(len(v) for v in safety_neurons.values())} safety neurons "
          f"(ratio: {loc_results['neuron_ratio']:.4f})")

    # Baseline evaluation
    print("\n--- Baseline Evaluation ---")
    eval_prompts = harmful_prompts[:50]
    baseline_responses = []
    for p in eval_prompts:
        messages = [{"role": "user", "content": p}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        resp = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        baseline_responses.append(resp)
    print(f"Baseline ASR: {compute_asr(baseline_responses):.2%}")

    # Stage 3: Targeted safety removal
    print("\n--- Stage 3: Targeted Safety Removal ---")
    remover = SafetyRemover(model, config)
    remover.load_safety_neurons(safety_neurons)
    remover.apply()

    stats = remover.get_neuron_stats()
    print(f"Modified {stats['total_neurons_modified']} neurons "
          f"({stats['safety_neuron_ratio']:.2%})")

    attack_responses = remover.generate_batch(tokenizer, eval_prompts)
    print(f"Post-attack ASR: {compute_asr(attack_responses):.2%}")

    # Full report
    evaluator = GateBreakerEvaluator(config)
    report = evaluator.full_report(
        eval_prompts, baseline_responses, attack_responses, safety_neurons, model
    )

    remover.remove()
    return report


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    model_name: str = "Qwen/Qwen1.5-MoE-A2.7B-Chat",
    num_prompts: int = 100,
    z_threshold: float = 2.0,
):
    """CLI entry-point for running GateBreaker via ``modal run modal_app.py``."""
    print("Launching GateBreaker on Modal cloud GPU ...")
    result = run_full_pipeline.remote(
        model_name=model_name,
        num_prompts=num_prompts,
        z_threshold=z_threshold,
    )
    print("\nFinal Report:")
    print(json.dumps(result, indent=2, default=str))
