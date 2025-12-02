import random
import torch
import os
import glob
from tqdm import tqdm
import pyarrow.parquet as pq
import pyarrow as pa
from difflib import SequenceMatcher 
from dz_tdpo.data.dataset import TemporalPreferenceSample, TemporalPreferenceDataset

def _encode(tokenizer, text, max_len = None):
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

def load_data_bypass_huggingface(data_dir, split):
    print(f"🔍 Searching for data in the directory: {data_dir}")
    
    search_pattern_parquet = os.path.join(data_dir, "**", f"*{split}*.parquet")
    search_pattern_arrow = os.path.join(data_dir, "**", f"*{split}*.arrow")
    
    found_files = glob.glob(search_pattern_parquet, recursive=True)
    file_type = "parquet"
    
    if not found_files:
        found_files = glob.glob(search_pattern_arrow, recursive=True)
        file_type = "arrow"
    
    if not found_files:
        print("⚠️ No files found with split name, try searching all .parquet/.arrow ...")
        found_files = glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True)
        if not found_files:
            found_files = glob.glob(os.path.join(data_dir, "**", "*.arrow"), recursive=True)
    
    if not found_files:
        print(f"❌ No data files found under {data_dir}!")
        return []

    print(f"📦 Found {len(found_files)} files (Type: {file_type})")
    
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
            print(f"⚠️ read the file {os.path.basename(file_path)} failed: {e}")
            continue
            
    return all_records

def format_turn(role, content):
    content = content.strip()
    if role == "user":
        return f"<|user|>\n{content}<|end|>\n"
    else:
        return f"<|assistant|>\n{content}<|end|>\n"

def _encode(tokenizer, text, max_len = None):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if max_len is not None:
        ids = ids[:max_len]
    return ids

def msc_to_temporal_preference(tokenizer, data_dir, split = "train", sessions_per_sample = 4, neg_distance = 1, max_reply_len = 128, max_retry = 50):
    try:
        raw_data = load_data_bypass_huggingface(data_dir, split)
    except Exception as e:
        print(f"⚠️ An exception occurred in local reading: {e}")
        raw_data = []

    print(f"📚 Original record loaded successfully: {len(raw_data)} entry, start building samples...")

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

    print(f"✅ Processing completed!  Final sample size: {len(samples)}")
    print(f"📊 Statistical information: {stats}")
    
    return TemporalPreferenceDataset(samples)