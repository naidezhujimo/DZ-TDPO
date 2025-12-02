from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from typing import List
from ..utils import compute_adaptive_tau

@dataclass
class TemporalPreferenceSample:
    context_ids: torch.Tensor
    context_turns: List[int]
    chosen_reply_ids: torch.Tensor
    rejected_reply_ids: torch.Tensor
    turn_id: int
    total_turns: int

class TemporalPreferenceDataset(Dataset):
    def __init__(self, samples, tokenizer, config):
        self.samples = samples
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        context_ignore = torch.full_like(sample.context_ids, -100)
        
        label_w = torch.cat([context_ignore, sample.chosen_reply_ids], dim=-1)
        label_l = torch.cat([context_ignore, sample.rejected_reply_ids], dim=-1)

        reply_start = len(sample.context_ids)
        chosen_boundaries = sample.context_turns + [reply_start]
        rejected_boundaries = sample.context_turns + [reply_start]
        if self.config.use_adaptive_tau:
            ada_tau = compute_adaptive_tau(
                self.tokenizer, 
                sample.context_ids, 
                self.config.tau
            )
        else:
            ada_tau = self.config.tau
        return {
            'input_ids': sample.context_ids,  # 用于可视化
            'labels_w': torch.cat([sample.context_ids, sample.chosen_reply_ids], dim=-1), # input
            'labels_l': torch.cat([sample.context_ids, sample.rejected_reply_ids], dim=-1), # input
            'masks_w': label_w, # [新增] 真正的 label，带 -100 mask
            'masks_l': label_l, # [新增] 真正的 label，带 -100 mask
            'turn_boundaries_w': chosen_boundaries,
            'turn_boundaries_l': rejected_boundaries,
            'turn_id': sample.turn_id,
            'total_turns': sample.total_turns,
            'adaptive_tau': float(ada_tau) # [新增] 返回 tau
        }