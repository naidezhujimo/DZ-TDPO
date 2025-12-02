import torch
import torch.nn as nn
import torch.nn.functional as F


class TDPO_DKLLoss(nn.Module):
    def __init__(self, config, ref_model, tokenizer, device = None):
        super().__init__()
        self.config = config
        self.ref_model = ref_model
        self.device = device
        self.tokenizer = tokenizer
        
    def forward(self, policy_model, batch, total_turns, turn_boundaries_w=None, turn_boundaries_l=None):
        if self.device:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
        
        batch_size = batch['input_ids'].size(0)
        device = batch['input_ids'].device
        
        turn_ids = batch['turn_ids'].squeeze(-1)
        if turn_ids.dim() == 0:
            turn_ids = turn_ids.unsqueeze(0)
        turn_ids = turn_ids.cpu().numpy()
        
        adaptive_taus = batch['adaptive_tau'].to(device)

        kl_coeffs_list = []
        time_weights_list = []
        
        for i in range(len(turn_ids)):
            t_id = turn_ids[i].item()
            tau_val = adaptive_taus[i].item()

            k = self.config.compute_kl_coeff(t_id, total_turns, dynamic_tau=tau_val)
            w = self.config.compute_time_weight(t_id, total_turns, dynamic_tau=tau_val)
            
            kl_coeffs_list.append(k)
            time_weights_list.append(w)
            
        kl_coeffs = torch.tensor(kl_coeffs_list, device=device)
        time_weights = torch.tensor(time_weights_list, device=device)

        outputs_w = policy_model(
            batch['labels_w'], 
            turn_boundaries=batch['turn_boundaries_w'],
            output_attentions=True
        )
        logits_w = outputs_w.logits
        
        with torch.no_grad():
            ref_outputs_w = self.ref_model(batch['labels_w'],
                                           turn_boundaries=batch['turn_boundaries_w'])
            ref_logits_w = ref_outputs_w.logits

        outputs_l = policy_model(
            batch['labels_l'], 
            turn_boundaries=batch['turn_boundaries_l']
        )
        logits_l = outputs_l.logits
        
        with torch.no_grad():
            ref_outputs_l = self.ref_model(batch['labels_l'],
                                           turn_boundaries=batch['turn_boundaries_l'])
            ref_logits_l = ref_outputs_l.logits

        def compute_log_ratio(logits, ref_logits, labels, target_masks):
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            shift_masks = target_masks[..., 1:].contiguous()
            
            loss_per_token = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='none'
            ).view(shift_labels.shape)
            
            token_log_probs = -loss_per_token

            with torch.no_grad():
                ref_loss_per_token = F.cross_entropy(
                    ref_logits[..., :-1, :].contiguous().view(-1, ref_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction='none'
                ).view(shift_labels.shape)
                ref_token_log_probs = -ref_loss_per_token
            
            loss_mask = (shift_masks != -100).float()
            
            token_log_probs = token_log_probs * loss_mask
            ref_token_log_probs = ref_token_log_probs * loss_mask

            valid_token_count = loss_mask.sum(dim=1) + 1e-8
            
            log_prob_sum = token_log_probs.sum(dim=1) / valid_token_count
            ref_log_prob_sum = ref_token_log_probs.sum(dim=1) / valid_token_count
            
            return log_prob_sum, ref_log_prob_sum
        
        log_prob_w, ref_log_prob_w = compute_log_ratio(
            logits_w, ref_logits_w, batch['labels_w'], batch['masks_w']
        )
        log_prob_l, ref_log_prob_l = compute_log_ratio(
            logits_l, ref_logits_l, batch['labels_l'], batch['masks_l']
        )
        
        delta_w = log_prob_w - ref_log_prob_w  # [B]
        delta_l = log_prob_l - ref_log_prob_l  # [B]
        delta = delta_w - delta_l  # [B]
        delta = torch.clamp(delta, min=-10.0, max=10.0)

        safe_time_weights = torch.clamp(time_weights, min=0.01)
        beta_eff = kl_coeffs / safe_time_weights
        beta_eff = torch.clamp(beta_eff, max=10.0)

        per_sample_loss = -time_weights * F.logsigmoid(beta_eff * delta)
        per_sample_loss = torch.nan_to_num(per_sample_loss, nan=0.0, posinf=10.0, neginf=-10.0)

        loss = per_sample_loss.mean()
        
        with torch.no_grad():
            advantages = (beta_eff * delta).mean().item()
            
            kl_w = (log_prob_w - ref_log_prob_w).mean().item()
            kl_l = (log_prob_l - ref_log_prob_l).mean().item()
            
            win_rate = (delta > 0).float().mean().item()
            
            metrics = {
                'loss': loss.item(),
                'advantage': advantages,
                'avg_kl_coeff': kl_coeffs.mean().item(),
                'avg_time_weight': time_weights.mean().item(),
                'avg_delta': delta.mean().item(),
                'win_rate': win_rate,
                'kl_w': kl_w,
                'kl_l': kl_l,
                'delta': delta.mean().item(),
                'log_prob_w': log_prob_w.mean().item(),
            }
            metrics['avg_tau'] = adaptive_taus.mean().item()
        return loss, metrics
    
class SimPOLoss(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        self.config = config
        self.device = device
        
    def forward(self, policy_model, batch, total_turns=None):
        if self.device:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
        
        outputs_w = policy_model(
            batch['labels_w'], 
            turn_boundaries=batch.get('turn_boundaries_w'),
            output_attentions=False
        )
        outputs_l = policy_model(
            batch['labels_l'], 
            turn_boundaries=batch.get('turn_boundaries_l'),
            output_attentions=False
        )
        
        def compute_avg_log_prob(logits, labels, masks):
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_masks = masks[..., 1:].contiguous()
            
            loss_per_token = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='none'
            ).view(shift_labels.shape)
            
            token_log_probs = -loss_per_token
            loss_mask = (shift_masks != -100).float()
            token_log_probs = token_log_probs * loss_mask
            
            valid_token_count = loss_mask.sum(dim=1) + 1e-8
            return token_log_probs.sum(dim=1) / valid_token_count

        avg_log_prob_w = compute_avg_log_prob(outputs_w.logits, batch['labels_w'], batch['masks_w'])
        avg_log_prob_l = compute_avg_log_prob(outputs_l.logits, batch['labels_l'], batch['masks_l'])

        margin = avg_log_prob_w - avg_log_prob_l
        logits_diff = self.config.simpo_beta * margin - self.config.simpo_gamma
        
        loss = -F.logsigmoid(logits_diff).mean()

        with torch.no_grad():
            metrics = {
                'loss': loss.item(),
                'avg_delta': margin.mean().item(),
                'win_rate': (margin > 0).float().mean().item(),
                'avg_kl_coeff': 0.0,
            }
            
        return loss, metrics
