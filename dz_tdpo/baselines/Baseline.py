import torch
from torch.utils.data import Dataset
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig, CPOTrainer, CPOConfig
from tqdm import tqdm
import random
import os
import glob
import pyarrow.parquet as pq
import pyarrow as pa
from difflib import SequenceMatcher 
from datasets import Dataset as HFDataset

from typing import List
from dataclasses import dataclass

# 全局变量
_SEMANTIC_MODEL = None

@dataclass
class TemporalPreferenceSample:
    context_ids: torch.Tensor
    context_turns: List[int]
    chosen_reply_ids: torch.Tensor
    rejected_reply_ids: torch.Tensor
    turn_id: int
    total_turns: int


def get_semantic_model(device='cpu'):
    global _SEMANTIC_MODEL
    if _SEMANTIC_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            print("Loading Semantic Embedding Model (all-MiniLM-L6-v2)...")
            _SEMANTIC_MODEL = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        except ImportError:
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")
    return _SEMANTIC_MODEL

def load_data_bypass_huggingface(data_dir, split):
    print(f"Searching for data in the directory: {data_dir}")
    
    search_pattern_parquet = os.path.join(data_dir, "**", f"*{split}*.parquet")
    search_pattern_arrow = os.path.join(data_dir, "**", f"*{split}*.arrow")
    
    found_files = glob.glob(search_pattern_parquet, recursive=True)
    file_type = "parquet"
    
    if not found_files:
        found_files = glob.glob(search_pattern_arrow, recursive=True)
        file_type = "arrow"
    
    if not found_files:
        print("No files found with split name, try searching all .parquet/.arrow ...")
        found_files = glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True)
        if not found_files:
            found_files = glob.glob(os.path.join(data_dir, "**", "*.arrow"), recursive=True)
    
    if not found_files:
        print(f"No data files found under {data_dir}!")
        return []

    print(f"Found {len(found_files)} files (Type: {file_type})")
    
    all_records = []
    for file_path in found_files:
        try:
            if file_type == "parquet":
                table = pq.read_table(file_path)
                all_records.extend(table.to_pylist())
            elif file_type == "arrow":
                try:
                    with pa.ipc.open_stream(file_path) as reader:
                        all_records.extend(reader.read_all().to_pylist())
                except:
                    with pa.ipc.open_file(file_path) as reader:
                        all_records.extend(reader.read_all().to_pylist())
        except Exception as e:
            print(f"read the file {os.path.basename(file_path)} failed: {e}")
            continue
            
    return all_records


def compute_adaptive_tau(tokenizer, input_ids, base_tau):
    try:
        from sentence_transformers import util
    except ImportError:
        return base_tau

    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()
        
    text = tokenizer.decode(input_ids, skip_special_tokens=True)
    
    if "<|user|>" in text:
        turns = text.split("<|user|>")
    elif "User:" in text:
        turns = text.split("User:")
    else:
        turns = text.split("\n")
    
    valid_turns = [t.strip() for t in turns if len(t.strip()) > 10]
    
    if len(valid_turns) < 2:
        return base_tau 
    
    current_turn = valid_turns[-1]
    history_turns = valid_turns[:-1]
    
    model = get_semantic_model()
    all_sentences = history_turns + [current_turn]
    embeddings = model.encode(all_sentences, convert_to_tensor=True, show_progress_bar=False)
    
    hist_embs = embeddings[:-1]
    curr_emb = embeddings[-1]
    
    cosine_scores = util.cos_sim(curr_emb, hist_embs)[0]
    max_sim = cosine_scores.max().item()
    
    if max_sim < 0.3: max_sim = 0.0
    
    gamma = 0.8 
    adaptive_tau = base_tau * (1.0 - gamma * max_sim)
    return max(0.5, adaptive_tau)

class TemporalPreferenceDataset(Dataset):
    def __init__(self, samples, tokenizer, config):
        self.samples = samples
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # 注意：TRL 不需要这个 __getitem__ 返回复杂的 tensor 字典
        # TRL 需要的是 prompt, chosen, rejected 的字符串
        # 这个类主要作为数据容器使用
        sample = self.samples[idx]
        return sample

def _encode(tokenizer, text, max_len = None):
    ids = tokenizer.encode(text, add_special_tokens=False) 
    vocab = tokenizer.vocab_size
    ids = [i if i < vocab else tokenizer.eos_token_id for i in ids]
    if max_len is not None:
        ids = ids[:max_len]
    return ids

def _encode_with_eos(tokenizer, text, max_len = None):
    ids = tokenizer.encode(text, add_special_tokens=False)
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 32007
    ids.append(eos_id)
    if max_len is not None and len(ids) > max_len:
        ids = ids[:max_len-1] + [eos_id]
    return ids

# --- 这里删除了重复的 load_data_bypass_huggingface ---

def format_turn(role, content):
    content = content.strip()
    if role == "user":
        return f"<|user|>\n{content}<|end|>\n"
    else:
        return f"<|assistant|>\n{content}<|end|>\n"

def msc_to_temporal_preference(tokenizer, data_dir, split = "train", sessions_per_sample = 4, neg_distance = 1, max_reply_len = 128, max_retry = 50):
    try:
        raw_data = load_data_bypass_huggingface(data_dir, split)
    except Exception as e:
        print(f"An exception occurred in local reading: {e}")
        raw_data = []

    print(f"Original record loaded successfully: {len(raw_data)} entry, start building samples...")

    samples = []
    stats = {
        "skipped_short": 0,
        "skipped_last_is_user": 0,
        "skipped_no_negative": 0,
        "success": 0
    }
    
    for record in tqdm(raw_data, desc=f"process MSC ({split})"):
        if "sessions" not in record: continue
            
        all_sessions = record["sessions"]
        if len(all_sessions) < sessions_per_sample:
            stats["skipped_short"] += 1
            continue
            
        start = random.randint(0, len(all_sessions) - sessions_per_sample)
        chunk = all_sessions[start: start + sessions_per_sample]

        dialogue_history = []
        
        for ses in chunk:
            if isinstance(ses, dict) and "dialogue" in ses:
                for i, utt in enumerate(ses["dialogue"]):
                    text = utt.get("text", "") if isinstance(utt, dict) else str(utt)
                    role = "user" if i % 2 == 0 else "assistant"
                    dialogue_history.append((role, text))

        if len(dialogue_history) < 10:
            stats["skipped_short"] += 1
            continue

        if dialogue_history[-1][0] == "user":
            dialogue_history = dialogue_history[:-1]
        
        if len(dialogue_history) < 10:
            stats["skipped_last_is_user"] += 1
            continue

        chosen_role, chosen_text = dialogue_history[-1]
        context_pairs = dialogue_history[:-1]     

        context_tokens = []
        boundaries = [0]
        
        for role, text in context_pairs:
            formatted_text = format_turn(role, text)
            ids = _encode(tokenizer, formatted_text)
            context_tokens.extend(ids)
            boundaries.append(len(context_tokens))
        trigger_text = "<|assistant|>\n"
        trigger_ids = tokenizer.encode(trigger_text, add_special_tokens=False)
        context_tokens.extend(trigger_ids)

        chosen_ids = _encode_with_eos(tokenizer, chosen_text, max_reply_len)

        valid_neg_indices = [
            i for i, (r, t) in enumerate(context_pairs) 
            if r == "assistant" and (len(context_pairs) - i) > neg_distance
        ]
        
        if not valid_neg_indices:
            stats["skipped_no_negative"] += 1
            continue

        found = False
        for _ in range(max_retry):
            neg_idx = random.choice(valid_neg_indices)
            neg_text = context_pairs[neg_idx][1]

            sim = SequenceMatcher(None, chosen_text, neg_text).ratio()
            if sim < 0.5:
                rejected_ids = _encode_with_eos(tokenizer, neg_text, max_reply_len)
                found = True
                break
        
        if not found:
            stats["skipped_no_negative"] += 1
            continue

        samples.append(TemporalPreferenceSample(
            context_ids        = torch.tensor(context_tokens, dtype=torch.long),
            context_turns      = boundaries,
            chosen_reply_ids   = torch.tensor(chosen_ids, dtype=torch.long),
            rejected_reply_ids = torch.tensor(rejected_ids, dtype=torch.long),
            turn_id            = len(boundaries) - 1,
            total_turns        = len(boundaries)
        ))
        stats["success"] += 1

    print(f"Processing completed!  Final sample size: {len(samples)}")
    print(f"Statistical information: {stats}")
    
    return TemporalPreferenceDataset(samples, tokenizer, None)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["dpo", "simpo"], help="Training mode")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to MSC dataset")
    parser.add_argument("--model_name", type=str, default="microsoft/Phi-3.5-mini-instruct")
    parser.add_argument("--output_dir", type=str, default="./baseline_checkpoints")
    return parser.parse_args()

def main():
    args = parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({'eos_token': '<|end|>'})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    special_tokens_dict = {'additional_special_tokens': ["<|user|>", "<|assistant|>", "<|end|>"]}
    tokenizer.add_special_tokens(special_tokens_dict)

    print("Loading MSC dataset using your original pipeline...")
    train_dataset = msc_to_temporal_preference(
        tokenizer,
        data_dir=args.data_dir,
        split="train",
        sessions_per_sample=4,
        neg_distance=5
    )
    val_dataset = msc_to_temporal_preference(
        tokenizer,
        data_dir=args.data_dir,
        split="validation",
        sessions_per_sample=4,
        neg_distance=5
    )

    print(f"Loaded {len(train_dataset)} samples.")

    def convert_dataset_to_trl_list(custom_dataset, tokenizer):
        trl_data = []
        for sample in tqdm(custom_dataset.samples, desc="Converting to TRL format"):
            trl_data.append({
                "prompt": tokenizer.decode(sample.context_ids, skip_special_tokens=False),
                "chosen": tokenizer.decode(sample.chosen_reply_ids, skip_special_tokens=False),
                "rejected": tokenizer.decode(sample.rejected_reply_ids, skip_special_tokens=False)
            })
        return trl_data

    print("Formatting train data for TRL...")
    train_data_list = convert_dataset_to_trl_list(train_dataset, tokenizer)
    
    print("Formatting val data for TRL...")
    val_data_list = convert_dataset_to_trl_list(val_dataset, tokenizer)
    
    trl_train_dataset = HFDataset.from_list(train_data_list)
    trl_val_dataset = HFDataset.from_list(val_data_list)
    
    model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"Loading Base Model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=model_dtype,
        attn_implementation="sdpa",
        device_map="auto"
    )
    model.resize_token_embeddings(len(tokenizer))

    if args.mode == "dpo":
        print("Starting Standard DPO Training...")
        training_args = DPOConfig(
            output_dir=f"{args.output_dir}/dpo",
            beta=0.1,
            learning_rate=1e-6, 
            per_device_train_batch_size=2,
            gradient_accumulation_steps=16,
            num_train_epochs=6,
            logging_steps=4,
            save_steps=50,
            save_total_limit=2,
            max_length=2400,
            max_prompt_length=3000,
            remove_unused_columns=False,
            gradient_checkpointing=True,
            bf16=True,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="rewards/accuracies",
        )
        
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=training_args,
            train_dataset=trl_train_dataset,
            eval_dataset=trl_val_dataset,
            processing_class=tokenizer,
        )

    elif args.mode == "simpo":
        print("Starting SimPO Training...")
        training_args = CPOConfig(
            output_dir=f"{args.output_dir}/simpo",
            loss_type="simpo",
            beta=2.0,
            simpo_gamma=1.4,
            learning_rate=5e-7,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=64,
            num_train_epochs=25,
            logging_steps=1,
            save_steps=50,
            max_length=2400,
            max_prompt_length=3000,
            remove_unused_columns=False,
            gradient_checkpointing=True,
            bf16=True,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="rewards/accuracies",
        )
        
        trainer = CPOTrainer(
            model=model,
            args=training_args,
            train_dataset=trl_train_dataset,
            eval_dataset=trl_val_dataset,
            processing_class=tokenizer,
        )

    print(f"Start training with {len(trl_train_dataset)} samples...")
    trainer.train()
    trainer.save_model(f"{args.output_dir}/{args.mode}_final")
    print("Training Finished!")

if __name__ == "__main__":
    main()