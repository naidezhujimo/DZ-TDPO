# benchmarks/eval_ppl.py
import argparse
import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dz_tdpo.config import TDPODKLConfig
from dz_tdpo.model import TemporalCausalLM
from dz_tdpo.data.dataset import TemporalPreferenceDataset
from dz_tdpo.data.msc import msc_to_temporal_preference

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="microsoft/Phi-3.5-mini-instruct")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the trained checkpoint (.pt)")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to MSC data directory")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--use_temporal_bias", action="store_true")
    return parser.parse_args()

def calculate_ppl(model, dataloader, device):
    model.eval()
    nlls = []
    token_counts = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating PPL"):
            if hasattr(model, "model"):
                input_ids = batch['labels_w'].to(device)
                labels = batch['masks_w'].to(device)

                outputs = model.model(input_ids=input_ids, labels=labels)
            else:
                outputs = model(
                    input_ids=batch['labels_w'].to(device),
                    labels=batch['masks_w'].to(device)
                )
            
            valid_tokens = (batch['masks_w'] != -100).sum().item()
            if valid_tokens > 0:
                nlls.append(outputs.loss.item() * valid_tokens)
                token_counts += valid_tokens

    return np.exp(np.sum(nlls) / token_counts)

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("🔄 Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    tokenizer.add_special_tokens({'additional_special_tokens': ["<|user|>", "<|assistant|>", "<|end|>"]})

    print(f"🔄 Loading Data from {args.data_dir}...")

    val_container = msc_to_temporal_preference(
        tokenizer, 
        data_dir=args.data_dir, 
        split="validation", 
        sessions_per_sample=4
    )
    

    dummy_config = TDPODKLConfig(use_adaptive_tau=False, tau=8.0) 
    val_dataset = TemporalPreferenceDataset(val_container.samples, tokenizer, dummy_config)
    
    def collate_fn(batch):
        from torch.nn.utils.rnn import pad_sequence
        labels_w = [b['labels_w'] for b in batch]
        masks_w = [b['masks_w'] for b in batch]
        
        padded_input = pad_sequence(labels_w, batch_first=True, padding_value=tokenizer.pad_token_id)
        padded_labels = pad_sequence(masks_w, batch_first=True, padding_value=-100)
        
        return {'labels_w': padded_input, 'masks_w': padded_labels}

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    print("\n🔵 Evaluating Base Model PPL...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto"
    )
    base_model.resize_token_embeddings(len(tokenizer))
    ppl_base = calculate_ppl(base_model, val_loader, device)
    del base_model
    torch.cuda.empty_cache()

    print(f"\n🟢 Evaluating Checkpoint: {args.ckpt_path}")
    config = TDPODKLConfig(model_name=args.model_name_or_path, use_temporal_bias=args.use_temporal_bias)
    
    base_for_ours = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto"
    )
    base_for_ours.resize_token_embeddings(len(tokenizer))
    
    policy_model = TemporalCausalLM(base_for_ours, config, device)
    policy_model.load_weights(args.ckpt_path)
    
    ppl_ours = calculate_ppl(policy_model, val_loader, device)

    print("\n" + "="*30)
    print(f"Base PPL: {ppl_base:.2f}")
    print(f"Ours PPL: {ppl_ours:.2f}")
    print(f"Delta   : {ppl_ours - ppl_base:+.2f}")
    print("="*30)

if __name__ == "__main__":
    main()