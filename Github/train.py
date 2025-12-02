import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from dz_tdpo.config import TDPODKLConfig
from dz_tdpo.model import TemporalCausalLM
from dz_tdpo.trainer import TDPODKLTrainer
from dz_tdpo.loss import SimPOLoss, TDPO_DKLLoss
from dz_tdpo.data.dataset import TemporalPreferenceDataset
from dz_tdpo.data.msc import msc_to_temporal_preference
from dz_tdpo.data.ultrachat import build_ultrachat_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to MSC/UltraChat dataset folder")
    
    parser.add_argument("--model_name_or_path", type=str, default="microsoft/Phi-3.5-mini-instruct")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")

    parser.add_argument("--loss_type", type=str, default="tdpo", choices=["tdpo", "simpo"])

    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1.5e-5)
    
    parser.add_argument("--use_temporal_bias", action="store_true")
    parser.add_argument("--use_adaptive_tau", action="store_true")
    
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(f"CUDA detected, using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Current GPU video memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("CUDA not detected, using CPU training")
    
    config = TDPODKLConfig(
        model_name=args.model_name_or_path,
        tau=args.tau,
        beta0=args.beta0,
        use_temporal_bias=args.use_temporal_bias,
        loss_type=args.loss_type
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({'eos_token': '<|end|>'})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    special_tokens_dict = {'additional_special_tokens': ["<|user|>", "<|assistant|>", "<|end|>"]}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<|end|>")
    print(f"Token increase the quantity: {num_added_toks}")

    model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"Load the model to the device, using dtype: {model_dtype}...")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=model_dtype,
        attn_implementation="sdpa"
    ).to(device)
    base_model.gradient_checkpointing_enable()
    policy_model = TemporalCausalLM(base_model, config, device)

    ref_model = None
    if args.loss_type == "tdpo":
        ref_base = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            dtype=model_dtype,
            attn_implementation="sdpa"
        ).to(device)

        if num_added_toks > 0:
            print(f"⚠️ Resizing model embeddings to {len(tokenizer)}")
            base_model.resize_token_embeddings(len(tokenizer))
            ref_base.resize_token_embeddings(len(tokenizer))

        ref_config = TDPODKLConfig(model_name=args.model_name_or_path, use_temporal_bias=False)
        ref_model = TemporalCausalLM(ref_base, ref_config, device)

    print("Loading MSC training set...")
    raw_train_ds = msc_to_temporal_preference(
        tokenizer,
        data_dir=args.data_dir,
        split="train",              
        sessions_per_sample=4,      
        neg_distance=5
    )
    raw_val_ds = msc_to_temporal_preference(
        tokenizer,
        data_dir=args.data_dir,
        split="validation",
        sessions_per_sample=4,
        neg_distance=5
    )

    train_dataset = TemporalPreferenceDataset(raw_train_ds.samples, tokenizer, config)
    val_dataset = TemporalPreferenceDataset(raw_val_ds.samples, tokenizer, config)

    print(f"MSC training samples：{len(train_dataset)}")
    print(f"MSC validation sample：{len(val_dataset)}")
    for i in range(min(5, len(train_dataset))):
        sample = train_dataset[i]
        for key in ['input_ids', 'chosen_reply_ids', 'rejected_reply_ids']:
            tensor = getattr(sample, key, None)
            if tensor is not None and tensor.max() >= tokenizer.vocab_size:
                print(f"sample {i} is {key} cross-border: {tensor.max()} >= {tokenizer.vocab_size}")

    trainer = TDPODKLTrainer(
        policy_model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        learning_rate=1.5e-5,
        device=device,
        gradient_accumulation_steps=8,
        output_dir=args.output_dir
    )

    trainer.train(num_epochs=args.epochs)

    trainer.save_checkpoint("final_model.pt")

if __name__ == "__main__":
    main()
