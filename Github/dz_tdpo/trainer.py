import torch
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from .loss import TDPO_DKLLoss, SimPOLoss
from .utils import split_subset_metrics

class TDPODKLTrainer:
    def __init__(self, policy_model, ref_model, tokenizer, config, train_dataset, val_dataset = None, learning_rate = 1e-5, device = None, gradient_accumulation_steps=4, output_dir="./checkpoints"):
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Use device: {self.device}")
        
        self.policy = policy_model.to(self.device)
        self.ref_model = ref_model.to(self.device)
        self.tokenizer = tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.config = config
        self.output_dir = output_dir
        
        for param in self.ref_model.parameters():
            param.requires_grad = False
        print("Policy trainable parameters:", sum(p.numel() for p in self.policy.parameters() if p.requires_grad))
        print("Ref trainable parameters:", sum(p.numel() for p in self.ref_model.parameters() if p.requires_grad))

        trainable_params = []
        matb_params = []
        
        for name, p in self.policy.named_parameters():
            if p.requires_grad:
                if "temporal_bias" in name:
                    matb_params.append(p)
                    print(f"🔍 Find DZ-TA parameters: {name}")
                else:
                    trainable_params.append(p)
        
        backbone_params = []
        tab_params = []
        
        print("Separating parameter groups...")
        for name, param in self.policy.named_parameters():
            if param.requires_grad:
                if "temporal_bias" in name:
                    tab_params.append(param)
                    print(f"  -> DZ-TA parameters: {name}")
                else:
                    backbone_params.append(param)
        
        optimizer_grouped_parameters = [
            {'params': backbone_params, 'lr': learning_rate}, 
            {'params': tab_params,      'lr': 1e-4} 
        ]
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            weight_decay=0.01
        )

        num_epochs = 4
        steps_per_epoch = len(train_dataset) // 2 // gradient_accumulation_steps
        total_steps = steps_per_epoch * num_epochs
        warmup_steps = int(total_steps * 0.1)

        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        if config.loss_type == "simpo":
            self.loss_fn = SimPOLoss(config, self.device)
        else:
            self.loss_fn = TDPO_DKLLoss(config, ref_model, tokenizer, self.device)
        
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=2, 
            shuffle=True,
            pin_memory=True if self.device.type == 'cuda' else False,
            collate_fn=self.collate_fn
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=2,
            shuffle=False,
            pin_memory=True if self.device.type == 'cuda' else False,
            collate_fn=self.collate_fn
        ) if val_dataset else None
        
        self.global_step = 0
        
        if self.device.type == 'cuda' and next(self.policy.parameters()).dtype == torch.float16:
            self.scaler = GradScaler('cuda')
        else:
            self.scaler = None
        self.interval_wins = []
        self.global_wins = []
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.train_step = self._train_step_with_scaler

    def collate_fn(self, batch):
            max_len_w = max(b['labels_w'].size(-1) for b in batch)
            max_len_l = max(b['labels_l'].size(-1) for b in batch)
            max_len = max(max_len_w, max_len_l, self.config.max_context_length)
            
            padded_batch = {
                'input_ids': [], 'labels_w': [], 'labels_l': [], 
                'masks_w': [], 'masks_l': [],
                'turn_ids': [],
                'turn_boundaries_w': [], 'turn_boundaries_l': [],
                'adaptive_tau': []
            }

            for b in batch:
                ctx = b['input_ids'][:self.config.max_context_length]
                pad_ctx = F.pad(ctx, (max_len - len(ctx), 0), value=self.tokenizer.pad_token_id)
                
                labels_w = b['labels_w'][:max_len]
                pad_len_w = max_len - len(labels_w)
                pad_w = F.pad(labels_w, (pad_len_w, 0), value=self.tokenizer.pad_token_id)
                
                labels_l = b['labels_l'][:max_len]
                pad_len_l = max_len - len(labels_l)
                pad_l = F.pad(labels_l, (pad_len_l, 0), value=self.tokenizer.pad_token_id)
                
                padded_batch['input_ids'].append(pad_ctx)
                padded_batch['labels_w'].append(pad_w)
                padded_batch['labels_l'].append(pad_l)
                padded_batch['turn_ids'].append(b['turn_id'])

                masks_w = b['masks_w'][:max_len]
                pad_len_w = max_len - len(masks_w)
                pad_mask_w = F.pad(masks_w, (pad_len_w, 0), value=-100)
                
                masks_l = b['masks_l'][:max_len]
                pad_len_l = max_len - len(masks_l)
                pad_mask_l = F.pad(masks_l, (pad_len_l, 0), value=-100)
                
                padded_batch['masks_w'].append(pad_mask_w)
                padded_batch['masks_l'].append(pad_mask_l)

                adjusted_boundaries_w = [min(bd + pad_len_w, max_len - 1) for bd in b['turn_boundaries_w']]
                adjusted_boundaries_l = [min(bd + pad_len_l, max_len - 1) for bd in b['turn_boundaries_l']]
                
                padded_batch['turn_boundaries_w'].append(adjusted_boundaries_w)
                padded_batch['turn_boundaries_l'].append(adjusted_boundaries_l)

                padded_batch['adaptive_tau'].append(b['adaptive_tau'])

            return {
                'input_ids': torch.stack(padded_batch['input_ids']),
                'labels_w': torch.stack(padded_batch['labels_w']),
                'labels_l': torch.stack(padded_batch['labels_l']),
                'masks_w': torch.stack(padded_batch['masks_w']),
                'masks_l': torch.stack(padded_batch['masks_l']),
                'turn_boundaries_w': padded_batch['turn_boundaries_w'],
                'turn_boundaries_l': padded_batch['turn_boundaries_l'],
                'turn_ids': torch.tensor(padded_batch['turn_ids'], dtype=torch.long),
                'adaptive_tau': torch.tensor(padded_batch['adaptive_tau'], dtype=torch.float)
            }

    def log_samples(self, batch, metrics, step):

            print(f"\n{'='*20} Step {step} Visual Monitoring {'='*20}")
            
            input_ids = batch['input_ids'][0]

            valid_input = input_ids[input_ids != self.tokenizer.pad_token_id]
            context_text = self.tokenizer.decode(valid_input, skip_special_tokens=True)
            
            valid_label_w = batch['labels_w'][0]
            valid_label_w = valid_label_w[valid_label_w != self.tokenizer.pad_token_id]
            chosen_text = self.tokenizer.decode(valid_label_w, skip_special_tokens=True)
            
            valid_label_l = batch['labels_l'][0]
            valid_label_l = valid_label_l[valid_label_l != self.tokenizer.pad_token_id]
            rejected_text = self.tokenizer.decode(valid_label_l, skip_special_tokens=True)
            
            print(f"【Context (Capture the first 100 words)】: {context_text[:100]}...")
            print(f"【Chosen (better)】: ...{chosen_text[-100:]}")
            print(f"【Rejected (inferior)】: ...{rejected_text[-100:]}")
            interval_wr = metrics.get('interval_win_rate', 0.0)
            print(f"【Scores】: Delta={metrics.get('avg_delta', 0):.2f} | Current Step Win={'YES' if metrics.get('win_rate', 0)>0 else 'NO'} | 🟢Last 80 steps win rate: {interval_wr:.1%}")
            
            print("【Model Generation Test】: generating...", end="", flush=True)
            self.policy.eval()
            with torch.no_grad():
                prompt_ids = input_ids[-256:].unsqueeze(0).to(self.device)
                attention_mask = (prompt_ids != self.tokenizer.pad_token_id).long()
                try:
                    gen_out = self.policy.generate(
                        input_ids=prompt_ids,
                        max_new_tokens=30,
                        attention_mask=attention_mask,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                    new_tokens = gen_out[0][prompt_ids.shape[1]:]
                    gen_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                    print(f"\r【Model Generation Test】: {gen_text}")
                except Exception as e:
                    print(f"\r【Model Generation Test】: generation failure ({e})")
            
            self.policy.train()
            print("="*60 + "\n")

    def _train_step_with_scaler(self, batch):
        self.policy.train()

        total_turns = batch['turn_ids'].max().item() + 1
        
        with autocast('cuda', enabled=self.scaler is not None):
            loss, metrics = self.loss_fn(self.policy, batch, total_turns)
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Step {self.global_step} loss is nan/inf, skip")
            return metrics
        
        loss = loss / self.gradient_accumulation_steps

        is_win = metrics.get('win_rate', 0.0)
        self.interval_wins.append(is_win)
        self.global_wins.append(is_win)
            
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        grad_norm = 0.0 

        if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            
            valid_len_w = (batch['masks_w'] != -100).sum(dim=1).float().mean().item()
            valid_len_l = (batch['masks_l'] != -100).sum(dim=1).float().mean().item()
            avg_delta = metrics.get('avg_delta', 0)
            
            if len(self.interval_wins) > 0:
                interval_acc = sum(self.interval_wins) / len(self.interval_wins)
            else:
                interval_acc = 0.0
            
            if len(self.global_wins) > 0:
                global_acc = sum(self.global_wins) / len(self.global_wins)
            else:
                global_acc = 0.0

            metrics['interval_win_rate'] = interval_acc
            metrics['global_win_rate'] = global_acc
            
            if (self.global_step + 1) % (10 *self.gradient_accumulation_steps) == 0:
                print(f"🔄 Step {self.global_step+1} Update | gradient: {grad_norm:.4f} | Delta: {avg_delta:.4f} | Len: {valid_len_w:.1f}")
                print(f"📊 Win rate statistics | recent: {interval_acc:.2%} | full: {global_acc:.2%}")

            if (self.global_step + 1) % (10 * self.gradient_accumulation_steps) == 0:
                    self.log_samples(batch, metrics, self.global_step)
                    self.interval_wins = []

            self.lr_scheduler.step()

        self.global_step += 1
        return metrics
    
    def evaluate(self, epoch):
        self.policy.eval()
        all_metrics = []
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False):
                total_turns = batch['turn_ids'].max().item() + 1
                _, metrics = self.loss_fn(self.policy, batch, total_turns)
                all_metrics.append(metrics)
        
        processed_metrics = []
        for m in all_metrics:
            item = {}
            for k, v in m.items():
                item[k] = v.cpu().item() if isinstance(v, torch.Tensor) else v
            processed_metrics.append(item)
            
        avg_metrics = split_subset_metrics(processed_metrics, [True]*len(processed_metrics))
        
        samples = self.val_loader.dataset.samples
        recent_mask = [s.turn_id < 3 for s in samples]
        long_mask = [s.turn_id > 6 for s in samples]
        
        if not any(long_mask): long_mask[0] = True
        
        rec_metrics = split_subset_metrics(processed_metrics, recent_mask)
        long_metrics = split_subset_metrics(processed_metrics, long_mask)
        
        print(f"\n=== Epoch {epoch} result ===")
        print(f"Overall win rate : {avg_metrics.get('win_rate', 0):.2%}")
        print(f"Long-term winning rate (Hard): {long_metrics.get('win_rate', 0):.2%}")
        print("=======================\n")
        
        avg_metrics['long_win_rate'] = long_metrics.get('win_rate', 0.0)
        return avg_metrics
    
    def train(self, num_epochs, save_every = 1000):
        print("Start TDPO-DKL Training...")
        print(f"configuration: beta0={self.config.beta0}, alpha={self.config.alpha_min}, tau={self.config.tau}")
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"🚀 Training started. Output dir: {self.output_dir}")
        
        best_long_wr = 0.0
        for epoch in range(num_epochs):
            epoch_metrics = []
            
            with tqdm(self.train_loader, desc=f"Epoch {epoch}/{num_epochs}", 
                    leave=False, ncols=100) as pbar:
                for batch_idx, batch in enumerate(pbar):
                    metrics = self.train_step(batch)
                    
                    pbar.set_postfix({
                        'loss': f"{metrics['loss']:.4f}",
                        'win': f"{metrics['win_rate']:.2%}",
                        'β': f"{metrics['avg_kl_coeff']:.4f}"
                    })

                    epoch_metrics.append(metrics)
                
                    if batch_idx % 25 == 0:
                        print(
                            f"Epoch {epoch} | Step {self.global_step} | "
                            f"Loss: {metrics['loss']:.4f} | "
                            f"Avg KL β: {metrics['avg_kl_coeff']:.4f} | "
                            f"Win Rate: {metrics['win_rate']:.2%} | "
                            f"Advantage: {metrics['advantage']:.4f}"
                            f"log_prob_w: {metrics['log_prob_w']:.2f}"
                            f"Delta: {metrics['delta']:.4f}"
                            f"Tau: {metrics['avg_tau']:.4f}"
                        )
                    
                    if self.global_step > 0 and self.global_step % save_every == 0:
                        save_path = os.path.join(self.output_dir, f"tdpo_dkl_final.pt")
                        self.save_checkpoint(save_path, save_optimizer=False)
                        print(f"\n[Checkpoint] ✅ The model has been saved to: {save_path}\n")
            
            if self.val_loader:
                val_metrics = self.evaluate(epoch)
                long_wr = val_metrics.get('long_win_rate', 0.0)
                
                if long_wr > best_long_wr:
                    print(f"🏆 New record! Long-term winning rate: {best_long_wr:.2%} -> {long_wr:.2%}")
                    best_long_wr = long_wr
                    self.save_checkpoint(os.path.join(self.output_dir, "tdpo_dkl_final.pt"))
                    
                self.save_checkpoint(os.path.join(self.output_dir, "tdpo_dkl_final.pt"))
            
            avg_epoch = {
                k: np.mean([m[k] for m in epoch_metrics])
                for k in epoch_metrics[0].keys()
            }
            print(f"Epoch {epoch} average: Loss={avg_epoch['loss']:.4f}, "
                  f"Win Rate={avg_epoch['win_rate']:.2%}")
    
    def save_checkpoint(self, filename, save_optimizer = False):
            if not os.path.isabs(filename):
                save_path = os.path.join(self.output_dir, filename)
            else:
                save_path = filename

            policy_state_dict = {
                k: v.cpu().to(torch.bfloat16) 
                for k, v in self.policy.state_dict().items()
            }
            
            save_dict = {
                'policy_state_dict': policy_state_dict,
                'config': self.config,
                'global_step': self.global_step
            }

            if save_optimizer:
                optimizer_state_dict = {
                    k: v.cpu() if isinstance(v, torch.Tensor) else v 
                    for k, v in self.optimizer.state_dict().items()
                }
                save_dict['optimizer_state_dict'] = optimizer_state_dict
            
            torch.save(save_dict, save_path)
            
            file_size = os.path.getsize(save_path) / (1024 * 1024)
            print(f"Checkpoint saved to {save_path} | Size: {file_size:.2f} MB")