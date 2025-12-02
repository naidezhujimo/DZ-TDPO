import torch
import torch.nn as nn
from torch.amp import autocast

class TemporalAttentionBias(nn.Module):
    def __init__(self, max_positions = 4096, lambda_init = 0.5, tau_fixed = 10.0, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        self.lambda_strength = nn.Parameter(torch.tensor(lambda_init, dtype=dtype))
        self.register_buffer("tau_fixed", torch.tensor(tau_fixed, dtype=dtype))
        self.use_shielding = False
        self.shield_length = 100

    def compute_bias(self, query_length, key_length, turn_boundaries, device):
        token_turn_ids = torch.zeros(key_length, dtype=torch.long, device=device)
        for turn_idx, start_pos in enumerate(turn_boundaries):
            end_pos = turn_boundaries[turn_idx + 1] if turn_idx + 1 < len(turn_boundaries) else key_length
            token_turn_ids[start_pos:end_pos] = turn_idx
            
        q_ids = token_turn_ids[:query_length].unsqueeze(1)
        k_ids = token_turn_ids[:key_length].unsqueeze(0)
        
        dist = (q_ids - k_ids).to(self.dtype)
        

        valid_history_mask = (dist > 0).float()

        dist_norm = dist / self.tau_fixed

        bias = -torch.abs(self.lambda_strength) * dist_norm * valid_history_mask
        
        bias = torch.clamp(bias, min=-100.0)
        
        return bias.unsqueeze(0).unsqueeze(0)
    
    def compute_bias_tensor(self, seq_len, device):
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        diff = positions.unsqueeze(1) - positions.unsqueeze(0)
        distance = torch.clamp(diff, min=0)
        
        bias = -self.lambda_strength * (distance / self.tau_fixed)
        
        if self.use_shielding and seq_len > self.shield_length:
            shield_mask = torch.arange(seq_len, device=device) < self.shield_length
            shield_mask = shield_mask.unsqueeze(0)
            
            bias = bias.masked_fill(shield_mask, 0.0)
            
        return bias.unsqueeze(0).unsqueeze(0)

    def compute_bias_for_batch(self, query_length, key_length, turn_boundaries_batch, device):
        batch_size = len(turn_boundaries_batch)
        result = torch.zeros(batch_size, 1, query_length, key_length, 
                            dtype=self.dtype, device=device)
        for i, boundaries in enumerate(turn_boundaries_batch):
            result[i] = self.compute_bias(query_length, key_length, boundaries, device).squeeze(0)
        return result
        
class TemporalCausalLM(nn.Module):
    def __init__(self, model, config, device = None):
        super().__init__()
        self.model = model
        self.device = device or model.device
        self.config = config

        self.model_dtype = next(model.parameters()).dtype

        self.temporal_bias = None
        if config.use_temporal_bias:
            self.temporal_bias = TemporalAttentionBias(
                max_positions=getattr(model.config, "max_position_embeddings", 4096),
                lambda_init=config.bias_lambda,
                tau_fixed=config.tau_fixed
            ).to(self.device)
            
        self._freeze_reference_params()
    
    def _freeze_reference_params(self):
        for name, param in self.model.named_parameters():
            if any(k in name for k in ['bias', 'layer_norm', 'layernorm']):
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask = None, turn_boundaries = None, output_attentions = False, **kwargs):
        input_ids = input_ids.to(self.device)
        batch_size, seq_len = input_ids.shape
        
        if self.temporal_bias is not None and turn_boundaries is not None:
            temporal_bias = self.temporal_bias.compute_bias_for_batch(
                seq_len, seq_len, turn_boundaries, self.device
            )
            temporal_bias = temporal_bias.to(self.model_dtype)

            if attention_mask is None:
                attention_mask = torch.ones((batch_size, seq_len), device=self.device)
            
            if attention_mask.dim() == 2:
                extended_mask = (1.0 - attention_mask) * torch.finfo(self.model_dtype).min
                extended_mask = extended_mask.unsqueeze(1).unsqueeze(1) # [B, 1, 1, L]
            else:
                extended_mask = attention_mask
            final_attention_mask = extended_mask + temporal_bias
            
        else:
            final_attention_mask = attention_mask
        
        with autocast('cuda', enabled=True):
            out = self.model(input_ids=input_ids,
                            attention_mask=final_attention_mask,
                            output_attentions=False,
                            **kwargs)
        return out
    
    def generate(self, input_ids,turn_boundaries = None, **kwargs):
        input_ids = input_ids.to(self.device)

        if self.temporal_bias is not None and turn_boundaries is not None:
            batch_size, seq_len = input_ids.shape
            temporal_bias = self.temporal_bias.compute_bias(
                seq_len, seq_len, turn_boundaries, self.device
            )

            temporal_bias = temporal_bias.to(self.model_dtype)
            kwargs['attention_mask'] = temporal_bias.squeeze(0).to(self.model_dtype)  # 移除batch维度
        
        with autocast('cuda', enabled=True):
            return self.model.generate(input_ids=input_ids, **kwargs)
        
    def load_weights(self, path):
        print(f"Loading Checkpoint: {path}")
        try:
            ckpt = torch.load(path, map_location=self.device)
            state_dict = ckpt['policy_state_dict'] if 'policy_state_dict' in ckpt else ckpt
            
            if self.temporal_bias:
                lambda_key = next((k for k in state_dict.keys() if "lambda_strength" in k), None)
                if lambda_key:
                    self.temporal_bias.lambda_strength.data = state_dict[lambda_key].data
                    print(f"Loaded lambda: {self.temporal_bias.lambda_strength.item()}")
                else:
                    print("Warning: lambda_strength not found in checkpoint.")

            self.load_state_dict(state_dict, strict=False)
            print("Model weights loaded.")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")