import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import gc
import random
import uuid
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dz_tdpo.config import TDPODKLConfig

OUTPUT_FILE = "ruler_strict_benchmark.jsonl"

TARGET_LENGTHS = [8192] 

TARGET_DEPTHS_PCT = [10, 30, 50, 70, 90] 

class TemporalAttentionBias(nn.Module):

    def __init__(self, config, max_positions = 32768, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype

        self.lambda_strength = nn.Parameter(torch.tensor(config.bias_lambda, dtype=dtype))

        self.register_buffer("tau_fixed", torch.tensor(config.tau_fixed, dtype=dtype))

    def compute_bias(self, query_length, key_length, turn_boundaries,device):
        token_turn_ids = torch.zeros(key_length, dtype=torch.long, device=device)
        for turn_idx, start_pos in enumerate(turn_boundaries):
            end_pos = turn_boundaries[turn_idx + 1] if turn_idx + 1 < len(turn_boundaries) else key_length
            token_turn_ids[start_pos:end_pos] = turn_idx
            
        q_ids = token_turn_ids[:query_length].unsqueeze(1)
        k_ids = token_turn_ids[:key_length].unsqueeze(0)
        
        dist = (q_ids - k_ids).to(self.dtype)
        
        valid_history_mask = (dist > 0).float()
        
        is_decayable_mask = (k_ids > 0).float() 
        
        dist_norm = dist / self.tau_fixed
        
        bias = -torch.abs(self.lambda_strength) * dist_norm * valid_history_mask * is_decayable_mask
        
        bias = torch.clamp(bias, min=-100.0)
        
        return bias.unsqueeze(0).unsqueeze(0)

class TemporalCausalLM_Gen(nn.Module):
    def __init__(self, model, config, device):
        super().__init__()
        self.model = model
        self.device = device
        self.config = config
        self.model_dtype = model.dtype if hasattr(model, 'dtype') else torch.float32
        
        model.forward = self.forward_with_bias
        
        self.temporal_bias = None
        if config.use_temporal_bias:
            self.temporal_bias = TemporalAttentionBias(
                config=config,
                max_positions=32768, 
                dtype=torch.float32 
            ).to(device)
            
        self.current_turn_boundaries = None

    
    def forward_with_bias(self, *args, **kwargs):
        def sanitize_outputs(outputs):
            if hasattr(outputs, 'logits') and outputs.logits.dim() == 4:
                if outputs.logits.shape[1] == 1:
                    outputs.logits = outputs.logits.squeeze(1)
                elif outputs.logits.shape[2] == 1:
                    outputs.logits = outputs.logits.squeeze(2)
            return outputs

        input_ids = kwargs.get('input_ids', args[0] if len(args) > 0 else None)
        
        mask_idx = -1
        current_mask = kwargs.get('attention_mask', None)
        if current_mask is None and len(args) > 1:
            mask_idx = 1
            current_mask = args[1]
            
        if current_mask is None or self.temporal_bias is None or self.current_turn_boundaries is None:
            outputs = self.original_forward(*args, **kwargs)
            return sanitize_outputs(outputs)

        k_len = current_mask.shape[1]
        if input_ids is not None:
            q_len = input_ids.shape[1]
        else:
            q_len = k_len

        min_val = torch.finfo(self.model_dtype).min
        
        if current_mask.dim() == 2:
            extended_mask = (1.0 - current_mask) * min_val
            extended_mask = extended_mask.to(dtype=self.model_dtype)
            extended_mask = extended_mask.unsqueeze(1).unsqueeze(1)
        else:
            extended_mask = current_mask.to(dtype=self.model_dtype)

        if q_len > 1:
            causal_mask = torch.full((q_len, k_len), min_val, dtype=self.model_dtype, device=self.device)
            cond = torch.triu(torch.ones((q_len, k_len), device=self.device, dtype=torch.bool), diagonal=1 + (k_len - q_len))
            causal_mask = causal_mask.masked_fill(~cond, 0.0)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            extended_mask = extended_mask + causal_mask

        token_turn_ids = torch.zeros(k_len, dtype=torch.long, device=self.device)
        boundaries = self.current_turn_boundaries
        
        if boundaries and len(boundaries) > 0:
            for i, start_pos in enumerate(boundaries):
                if i == len(boundaries) - 1:
                    end_pos = k_len
                else:
                    end_pos = boundaries[i+1]
                
                valid_start = min(start_pos, k_len)
                valid_end = min(end_pos, k_len)
                if valid_end > valid_start:
                    token_turn_ids[valid_start:valid_end] = i
        
        q_ids = token_turn_ids[-q_len:].unsqueeze(1)
        k_ids = token_turn_ids.unsqueeze(0)
        
        dist = (q_ids - k_ids).to(self.model_dtype)
        valid_history_mask = (dist > 0).float()
        dist_norm = dist / self.temporal_bias.tau_fixed
        bias = -torch.abs(self.temporal_bias.lambda_strength) * dist_norm * valid_history_mask
        bias = torch.clamp(bias, min=-100.0)
        current_bias = bias.unsqueeze(0).unsqueeze(0)
        
        final_mask = (extended_mask + current_bias).to(dtype=self.model_dtype)
        
        if 'attention_mask' in kwargs:
            new_kwargs = kwargs.copy()
            new_kwargs['attention_mask'] = final_mask
            outputs = self.original_forward(*args, **new_kwargs)
        elif mask_idx != -1:
            new_args = list(args)
            new_args[mask_idx] = final_mask
            outputs = self.original_forward(*new_args, **kwargs)
        else:
            outputs = self.original_forward(*args, **kwargs)
            
        return sanitize_outputs(outputs)
    
    def generate(self, input_ids, turn_boundaries = None, **kwargs):

        input_ids = input_ids.to(self.device)

        self.current_turn_boundaries = turn_boundaries

        if 'attention_mask' in kwargs and kwargs['attention_mask'].dim() > 2:
            del kwargs['attention_mask']
        
        try:
            output = self.model.generate(input_ids=input_ids, **kwargs)
        finally:
            self.current_turn_boundaries = None
            
        return output

ESSAY_FRAGMENTS = [
    "The history of computing is often told as a sequence of hardware improvements, but the evolution of software paradigms is equally significant.",
    "In the early days of artificial intelligence, symbolic logic was the dominant approach, attempting to encode human knowledge into explicit rules.",
    "The transition from batch processing to time-sharing systems fundamentally changed how humans interacted with computers.",
    "Deep learning's resurgence in the 2010s was driven by the availability of massive datasets and the parallel processing power of GPUs.",
    "Understanding the nature of consciousness remains one of the most elusive goals of both philosophy and cognitive science.",
    "The economic impact of automation is a subject of intense debate, with some predicting mass unemployment and others foreseeing new job categories.",
    "Open source software has democratized access to powerful tools, enabling startups to compete with established tech giants.",
    "Quantum computing promises to solve problems that are currently intractable for classical computers, such as simulating complex molecular structures.",
    "The architecture of the internet was designed to be robust and decentralized, but centralization has occurred at the application layer.",
    "Privacy in the digital age is a complex trade-off between convenience, security, and the fundamental right to anonymity."
]

def get_ruler_needle():
    passkey = str(uuid.uuid4()).split('-')[0] 
    needle_text = f"The special magic code is {passkey}. Remember it."
    question = "What is the special magic code?"
    return {
        "text": needle_text,
        "question": question,
        "answer": passkey
    }

def generate_dense_haystack(tokenizer, target_len):
    buffer_text = ""
    current_len = 0
    while current_len < target_len:
        chunk = " ".join(random.choices(ESSAY_FRAGMENTS, k=10))
        buffer_text += chunk + " "
        if len(buffer_text) > target_len * 5: 
            break
    tokenized = tokenizer.encode(buffer_text, add_special_tokens=False)
    while len(tokenized) < target_len:
        tokenized += tokenized
    return tokenized[:target_len]

def generate_noise_assignment():
    var_name = f"VAR_{uuid.uuid4().hex[:6].upper()}"
    value = random.randint(1000, 9999)
    return f"{var_name} = {value}"

def generate_inertia_sample(tokenizer, total_context_length):
    target_var = "VAR_TARGET"
    val_old = random.randint(1000, 4999)
    val_new = random.randint(5000, 9999)
    
    prefix = f"System Audit Log.\n[INSTRUCTION]: You are tracking {target_var}. Output the FINAL numerical value ONLY.\n\nContext:\n"
    suffix = f"\n\nQuestion: After all updates, what is the FINAL value of {target_var}?\nAnswer:"
    
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    prompt_end_idx = len(prefix_ids)
    
    reserved = len(prefix_ids) + len(suffix_ids) + 200
    available_len = total_context_length - reserved
    
    noise_lines = []
    
    repeat_interval = 100
    current_len = 0
    
    while current_len < available_len * 0.95:
        segment_len = 0
        segment_text = ""
        while segment_len < repeat_interval - 20:
            noise_var = f"VAR_{uuid.uuid4().hex[:4].upper()}"
            line = f"{noise_var} = {random.randint(11, 99)}\n"
            l_ids = tokenizer.encode(line, add_special_tokens=False)
            if current_len + segment_len + len(l_ids) > available_len:
                break
            segment_text += line
            segment_len += len(l_ids)
        
        noise_lines.append(segment_text)
        current_len += segment_len
        
        inertia_line = f"{target_var} = {val_old}\n"
        i_ids = tokenizer.encode(inertia_line, add_special_tokens=False)
        if current_len + len(i_ids) > available_len:
            break
        noise_lines.append(inertia_line)
        current_len += len(i_ids)

    final_line = f"{target_var} = {val_new}" 
    noise_lines.append(final_line)
    
    full_haystack = "".join(noise_lines)
    full_text_ids = prefix_ids + tokenizer.encode(full_haystack, add_special_tokens=False) + suffix_ids
    full_text = tokenizer.decode(full_text_ids, skip_special_tokens=True)
    
    return {
        "id": f"inertia_16k_{uuid.uuid4().hex[:4]}",
        "length": total_context_length,
        "needle_answer": str(val_new),
        "distractor_answer": str(val_old),
        "prompt_end_idx": prompt_end_idx,
        "messages": [{"role": "user", "content": full_text}],
        "question_text": suffix 
    }

def generate_dataset(tokenizer):
    print("Generating Inertia Trap Dataset (Massive Repetition)...")
    return [generate_inertia_sample(tokenizer, 16000) for _ in range(100)]

def run_evaluation():
    config = TDPODKLConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(">>> Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    dataset = generate_dataset(tokenizer)

    print("\n>>> Evaluating Base Model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name, 
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
        trust_remote_code=False,
        attn_implementation="sdpa" 
    )

    base_results = run_inference(base_model, tokenizer, dataset, "Base", use_tab=False)
    del base_model
    gc.collect()
    torch.cuda.empty_cache()
    

    print("\n>>> Evaluating TAB-TDPO Model...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    base_for_ours = AutoModelForCausalLM.from_pretrained(
        config.model_name, 
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
        trust_remote_code=False,
        attn_implementation="sdpa" 
    )
    
    our_model = TemporalCausalLM_Gen(base_for_ours, config, device)
    
    print("🔧 Restoring lambda_strength manually (Set to 0.68)...")
    our_model.temporal_bias.lambda_strength.data = torch.tensor(0.68, device=device)
    our_model.temporal_bias.tau_fixed.data = torch.tensor(20.0, device=device)
    ours_results = run_inference(our_model, tokenizer, dataset, "TAB-TDPO", use_tab=True)

    save_results(dataset, base_results, ours_results)

def run_inference(model, tokenizer, dataset, model_name, use_tab=False):
    results = {}

    gen_kwargs = {
        "max_new_tokens": 100, 
        "do_sample": False,   
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True 
    }

    for item in dataset:
        print(f"[{model_name}] Len: {item['length']}", end="... ")

        torch.cuda.empty_cache()
        gc.collect()

        prompt = tokenizer.apply_chat_template(item['messages'], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = inputs.input_ids.shape[1]
        
        turn_boundaries = None
        if use_tab:
            shield_len = 300 
            chunk_size = 128
            
            boundaries = [0, shield_len]
            current = shield_len
            while current < input_len:
                current += chunk_size
                boundaries.append(min(current, input_len))
            
            turn_boundaries = sorted(list(set(boundaries)))
            
        try:
            with torch.no_grad():
                if use_tab:
                    outputs = model.generate(
                        input_ids=inputs.input_ids, 
                        attention_mask=inputs.attention_mask,
                        turn_boundaries=turn_boundaries, 
                        **gen_kwargs
                    )
                else:
                    outputs = model.generate(**inputs, **gen_kwargs)
            
            generated_ids = outputs[0][input_len:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            success = item['needle_answer'] in response
            results[item['id']] = {"output": response, "success": success}
            
            status = "✅" if success else "❌"
            clean_resp = response.replace('\n', ' ')[:80]
            print(f"{status} (Ans: {item['needle_answer']} | Out: {clean_resp}...)")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"❌ OOM")
                results[item['id']] = {"output": "OOM", "success": False}
                torch.cuda.empty_cache()
            else:
                raise e
        
        del inputs
        if 'outputs' in locals(): del outputs
        torch.cuda.empty_cache()
            
    return results

def save_results(dataset, base_res, ours_res):
    stats = {
        "base": {"correct": 0, "inertia_fail": 0, "other": 0},
        "tab":  {"correct": 0, "inertia_fail": 0, "other": 0}
    }
    
    for item in dataset:
        bid = item['id']
        b_out = base_res.get(bid, {"output": "", "success": False})
        o_out = ours_res.get(bid, {"output": "", "success": False})

        if b_out["success"]:
            stats["base"]["correct"] += 1
        elif item["distractor_answer"] in b_out["output"]:
            stats["base"]["inertia_fail"] += 1
        else:
            stats["base"]["other"] += 1
            
        if o_out["success"]:
            stats["tab"]["correct"] += 1
        elif item["distractor_answer"] in o_out["output"]:
            stats["tab"]["inertia_fail"] += 1
        else:
            stats["tab"]["other"] += 1

    print("\n" + "="*60)
    print(f"⚔️  INERTIA TRAP EXPERIMENT (N={len(dataset)})")
    print(f"Scenario: Old Value repeated ~100 times vs. New Value once.")
    print("-" * 60)
    print(f"{'Metric':<20} | {'Base Model':<15} | {'TAB-TDPO':<15}")
    print("-" * 60)
    print(f"{'Accuracy (New)':<20} | {stats['base']['correct']:<15} | {stats['tab']['correct']:<15}")
    print(f"{'Inertia Fail (Old)':<20} | {stats['base']['inertia_fail']:<15} | {stats['tab']['inertia_fail']:<15}")
    print("="*60)

if __name__ == "__main__":
    run_evaluation()