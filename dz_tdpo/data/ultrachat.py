import torch
from datasets import load_dataset
from tqdm import tqdm
import random
import os
import glob
import numpy as np
from .dataset import TemporalPreferenceSample 

def build_ultrachat_dataset(tokenizer, data_dir, num_samples=500, split="train_sft"):
    print(f"Loading UltraChat from: {data_dir} (Split: {split})")
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}. Please set --data_dir correctly.")

    search_pattern = os.path.join(data_dir, "**", f"*{split}*.parquet")
    found_files = glob.glob(search_pattern, recursive=True)
    
    if not found_files:
        print(f"No specific split file found. Searching all .parquet files...")
        found_files = glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True)
        
    if not found_files:
        raise FileNotFoundError(f"No .parquet files found in {data_dir}")
    
    print(f"Found {len(found_files)} files. Loading...")
    
    ds = load_dataset(
        "parquet", 
        data_files=found_files,
        split="train", 
        streaming=True
    )
    
    samples = []
    max_seq_len = 4096
    
    stats = {
        "scanned": 0,
        "skip_short_turn": 0,
        "skip_role_error": 0,
        "skip_no_history": 0,
        "skip_too_long": 0,
        "success": 0
    }

    try:
        end_token_id = tokenizer.convert_tokens_to_ids("<|end|>")
        if end_token_id is None:
            end_token_id = tokenizer.encode("<|end|>", add_special_tokens=False)[0]
    except:
        end_token_id = 32007 

    pbar = tqdm(ds, desc=f"Screening short samples", total=num_samples)
    
    for data in pbar:
        stats["scanned"] += 1
        
        if len(samples) >= num_samples:
            break
            
        messages = data['messages'] 
        
        if len(messages) < 4: 
            stats["skip_short_turn"] += 1
            continue
            
        last_msg = messages[-1]
        if last_msg['role'] != 'assistant':
            stats["skip_role_error"] += 1
            continue 
            
        chosen_text = last_msg['content']
        context_msgs = messages[:-1]
        
        try:
            assist_indices = [i for i, m in enumerate(context_msgs) if m['role'] == 'assistant']
            if len(assist_indices) < 1:
                stats["skip_no_history"] += 1
                continue
            
            if len(assist_indices) > 1 and random.random() < 0.5:
                rejected_idx = assist_indices[-1]
            else:
                rejected_idx = random.choice(assist_indices)
                
            rejected_text = context_msgs[rejected_idx]['content']
            
            if len(rejected_text) < 10 or abs(len(rejected_text) - len(chosen_text)) > 500:
                 continue

            if rejected_text == chosen_text:
                continue
        except Exception:
            continue

        chosen_ids = tokenizer.encode(chosen_text + tokenizer.eos_token, add_special_tokens=False)
        rejected_ids = tokenizer.encode(rejected_text + tokenizer.eos_token, add_special_tokens=False)
        reply_len = max(len(chosen_ids), len(rejected_ids))
        
        total_chars = sum(len(m['content']) for m in context_msgs)
        if total_chars > max_seq_len * 4: 
            stats["skip_too_long"] += 1
            continue

        try:
            full_context_ids = tokenizer.apply_chat_template(
                context_msgs, 
                tokenize=True, 
                add_generation_prompt=True
            )
        except Exception:
            continue
            
        if len(full_context_ids) + reply_len > max_seq_len:
            stats["skip_too_long"] += 1
            continue

        try:
            arr_ids = np.array(full_context_ids)
            end_positions = np.where(arr_ids == end_token_id)[0]
            
            fast_boundaries = [0]
            for pos in end_positions:
                if pos + 1 < len(full_context_ids):
                    fast_boundaries.append(int(pos + 1))
            
            if len(fast_boundaries) < len(context_msgs):
                 continue 

        except Exception:
            continue

        current_turn_id = len(context_msgs)

        samples.append(TemporalPreferenceSample(
            context_ids      = torch.tensor(full_context_ids, dtype=torch.long),
            context_turns    = fast_boundaries[:current_turn_id],
            chosen_reply_ids = torch.tensor(chosen_ids, dtype=torch.long),
            rejected_reply_ids = torch.tensor(rejected_ids, dtype=torch.long),
            turn_id          = current_turn_id,
            total_turns      = current_turn_id + 1 
        ))
        
        stats["success"] += 1
        
        if stats["scanned"] % 1000 == 0:
            pbar.set_postfix(hit_rate=f"{stats['success']/stats['scanned']:.1%}")

    print(f"\nConstruct statistics: {stats}")

    return samples
