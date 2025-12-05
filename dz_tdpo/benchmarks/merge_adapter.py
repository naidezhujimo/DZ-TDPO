import argparse
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dz_tdpo.config import TDPODKLConfig
from dz_tdpo.model import TemporalCausalLM

def parse_args():
    parser = argparse.ArgumentParser(description="Merge DZ-TDPO weights into Base Model")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to original HF model")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to training checkpoint (.pt)")
    parser.add_argument("--save_path", type=str, required=True, help="Output directory for merged model")
    return parser.parse_args()

def merge_and_save():
    args = parse_args()
    print(f"Loading Base Model from {args.base_model_path}...")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu", 
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)

    print("Resizing token embeddings...")
    tokenizer.add_special_tokens({'additional_special_tokens': ["<|user|>", "<|assistant|>", "<|end|>"]})
    model.resize_token_embeddings(len(tokenizer))

    print(f"Loading Checkpoint from {args.ckpt_path}...")
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    
    if isinstance(ckpt, dict) and 'policy_state_dict' in ckpt:
        state_dict = ckpt['policy_state_dict']
    else:
        state_dict = ckpt

    backbone_state_dict = {}
    custom_params = {}
    
    print("Cleaning state dict...")
    for k, v in state_dict.items():
        if "temporal_bias" in k or "lambda_strength" in k:
            custom_params[k] = v
            continue
            
        new_k = k
        if k.startswith("model.model."):
            new_k = k.replace("model.model.", "model.")
        elif k.startswith("model."):
            new_k = k.replace("model.", "") 
            
        backbone_state_dict[new_k] = v

    if custom_params:
        print("\nFound Custom Temporal Parameters:")
        for k, v in custom_params.items():
            print(f"   - {k}: {v.item() if v.numel() == 1 else v.shape}")
        
        custom_path = os.path.join(args.save_path, "temporal_bias.pt")
        if not os.path.exists(args.save_path): os.makedirs(args.save_path)
        torch.save(custom_params, custom_path)
        print(f"   -> Saved custom params to {custom_path}")

    print("mg Merging Backbone weights...")
    missing, unexpected = model.load_state_dict(backbone_state_dict, strict=False)
    
    print(f"   Missing keys: {len(missing)}")
    if len(missing) > 0: print(f"   Example: {missing[:3]}")
    print(f"   Unexpected keys: {len(unexpected)}")

    print(f"Saving merged HF model to {args.save_path}...")
    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    print("Merge Complete! You can now load this path directly with AutoModel.")

if __name__ == "__main__":

    merge_and_save()
