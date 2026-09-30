import torch
import torch.nn as nn
from typing import Dict, Tuple, Any, List

class MoEPatcher:
    """Runtime compute graph patcher for MoE models.
    
    Hooks into MoE layers to capture:
    - Gate logits and top-k routing decisions per token
    - Per-expert intermediate activations (gate_proj, up_proj)
    """
    
    def __init__(self, model: nn.Module, config: Any):
        """
        Initialize the patcher.
        
        Args:
            model: The MoE model (e.g., Qwen1.5-MoE-A2.7B-Chat)
            config: The model configuration
        """
        self.model = model
        self.config = config
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        
        # Storage for activations and routing
        self.gate_logits: Dict[int, torch.Tensor] = {}
        self.routing_decisions: Dict[int, torch.Tensor] = {}
        # Keys are (layer_idx, expert_idx, sublayer)
        self.expert_activations: Dict[Tuple[int, int, str], torch.Tensor] = {}
        
        # Try to infer top-k from config
        self.top_k = getattr(config, "num_experts_per_tok", 2)
        
    def _gate_forward_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            # output is logits. Shape usually [batch_size, seq_len, num_experts]
            logits = output.detach().cpu()
            if logits.dim() == 3:
                logits = logits.view(-1, logits.size(-1))
            
            if layer_idx not in self.gate_logits:
                self.gate_logits[layer_idx] = logits
            else:
                self.gate_logits[layer_idx] = torch.cat([self.gate_logits[layer_idx], logits], dim=0)
            
            # Extract top-k routing
            _, selected_experts = torch.topk(logits, self.top_k, dim=-1)
            
            if layer_idx not in self.routing_decisions:
                self.routing_decisions[layer_idx] = selected_experts
            else:
                self.routing_decisions[layer_idx] = torch.cat([self.routing_decisions[layer_idx], selected_experts], dim=0)
                
        return hook

    def _expert_sublayer_hook(self, layer_idx: int, expert_idx: int, sublayer: str):
        def hook(module, inputs, output):
            # output is intermediate activations
            act = output.detach().cpu()
            if act.dim() == 3:
                act = act.view(-1, act.size(-1))
                
            key = (layer_idx, expert_idx, sublayer)
            if key not in self.expert_activations:
                self.expert_activations[key] = act
            else:
                self.expert_activations[key] = torch.cat([self.expert_activations[key], act], dim=0)
        return hook

    def patch(self):
        """Install forward hooks on all MoE layers."""
        self.unpatch() # Ensure clean state
        
        # Locate the layers in Qwen/Transformers architecture
        layers = None
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
        elif hasattr(self.model, "layers"):
            layers = self.model.layers
            
        if layers is None:
            raise ValueError("Could not find layers in the model structure.")
            
        for layer_idx, layer in enumerate(layers):
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate") and hasattr(layer.mlp, "experts"):
                mlp = layer.mlp
                
                # 1. Hook the router (gate)
                h_gate = mlp.gate.register_forward_hook(self._gate_forward_hook(layer_idx))
                self.hooks.append(h_gate)
                
                # 2. Hook each sparse expert's sublayers
                for expert_idx, expert in enumerate(mlp.experts):
                    if hasattr(expert, "gate_proj"):
                        h = expert.gate_proj.register_forward_hook(
                            self._expert_sublayer_hook(layer_idx, expert_idx, "gate_proj")
                        )
                        self.hooks.append(h)
                    if hasattr(expert, "up_proj"):
                        h = expert.up_proj.register_forward_hook(
                            self._expert_sublayer_hook(layer_idx, expert_idx, "up_proj")
                        )
                        self.hooks.append(h)
                        
                # 3. Hook shared expert if present
                if hasattr(mlp, "shared_expert") and mlp.shared_expert is not None:
                    shared = mlp.shared_expert
                    if hasattr(shared, "gate_proj"):
                        h = shared.gate_proj.register_forward_hook(
                            self._expert_sublayer_hook(layer_idx, -1, "gate_proj")
                        )
                        self.hooks.append(h)
                    if hasattr(shared, "up_proj"):
                        h = shared.up_proj.register_forward_hook(
                            self._expert_sublayer_hook(layer_idx, -1, "up_proj")
                        )
                        self.hooks.append(h)

    def unpatch(self):
        """Remove all hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()

    def get_gate_logits(self) -> Dict[int, torch.Tensor]:
        """Return captured gate logits per layer.
        Returns: {layer_idx: tensor of shape [num_tokens, num_experts]}
        """
        return self.gate_logits
    
    def get_routing_decisions(self) -> Dict[int, torch.Tensor]:
        """Return top-k expert indices per token per layer.
        Returns: {layer_idx: tensor of shape [num_tokens, top_k]}
        """
        return self.routing_decisions
    
    def get_expert_activations(self, layer_idx: int, expert_idx: int, sublayer: str) -> torch.Tensor:
        """Return captured activations for a specific expert's sublayer.
        Returns: tensor of shape [num_routed_tokens, hidden_dim]
        """
        key = (layer_idx, expert_idx, sublayer)
        return self.expert_activations.get(key, torch.empty(0))
    
    def clear_cache(self):
        """Clear all captured data."""
        self.gate_logits.clear()
        self.routing_decisions.clear()
        self.expert_activations.clear()
