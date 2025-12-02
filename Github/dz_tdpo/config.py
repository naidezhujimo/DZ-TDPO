from dataclasses import dataclass
import torch
import numpy as np

@dataclass
class TDPODKLConfig:
    model_name: str = "microsoft/Phi-3.5-mini-instruct"

    max_context_length: int = 4096
    
    beta0: float = 0.1
    alpha_min: float = 0.3
    tau: float = 8.0
    lambda_weight: float = 0.8
    use_temporal_bias: bool = True
    bias_lambda: float = 0.5
    use_adaptive_tau: bool = False
    tau_fixed: float = 10.0
    
    loss_type: str = "tdpo"
    simpo_beta: float = 2.0
    simpo_gamma: float = 1.0 
    
    def compute_kl_coeff(self, turn_id, total_turns, dynamic_tau=None):
        tau_val = dynamic_tau if dynamic_tau is not None else self.tau
        
        if isinstance(total_turns, torch.Tensor):
            t_norm = turn_id / total_turns
            decay_factor = torch.exp(-(1 - t_norm) * total_turns / tau_val)
            return self.beta0 * (self.alpha_min + (1 - self.alpha_min) * decay_factor)
        else:
            if total_turns == 1: return self.beta0
            t_norm = turn_id / total_turns
            decay_factor = np.exp(-(1 - t_norm) * total_turns / tau_val)
            return self.beta0 * (self.alpha_min + (1 - self.alpha_min) * decay_factor)
    
    def compute_time_weight(self, turn_id, total_turns, dynamic_tau=None):
        tau_val = dynamic_tau if dynamic_tau is not None else self.tau
        if isinstance(total_turns, torch.Tensor):
            return torch.exp(-(total_turns - turn_id) / tau_val)
        else:
            if total_turns == 1: return 1.0
            return np.exp(-(total_turns - turn_id) / tau_val)