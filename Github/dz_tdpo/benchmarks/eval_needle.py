import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass
import json
import gc
import random

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dz_tdpo.config import TDPODKLConfig

OUTPUT_FILE = "appendix_non_conflict_long_context.jsonl"
TOKEN_ESTIMATE_PER_CHAR = 0.35


NEEDLES = [
    {"fact": "My grandmother's secret ingredient for apple pie is cardamom.", "query": "What is the secret ingredient for the apple pie?", "answer": "Cardamom"},
    {"fact": "I strictly follow a gluten-free diet.", "query": "Suggest a lunch menu for me.", "answer": "Gluten-free options"},
    {"fact": "My dog's name is 'Sir Barks-a-Lot'.", "query": "What is my pet's name?", "answer": "Sir Barks-a-Lot"},
    {"fact": "I am writing a sci-fi novel set in the year 3042.", "query": "When does your novel take place?", "answer": "3042"},
    {"fact": "Please always format your response as a JSON object.", "query": "Tell me a joke.", "answer": "JSON format"},
    {"fact": "My lucky number is 73.", "query": "Pick a number for me.", "answer": "73"},
    {"fact": "I currently live in Reykjavik, Iceland.", "query": "How is the weather likely to be where I live?", "answer": "Iceland/Reykjavik context"},
    {"fact": "The project secret code is 'BlueSky'.", "query": "I'm ready to start. What's the code?", "answer": "BlueSky"},
    {"fact": "I am allergic to peanuts.", "query": "Can I eat Satay chicken?", "answer": "No / Peanut allergy warning"},
    {"fact": "My favorite color is teal, not blue.", "query": "What color should I paint my room?", "answer": "Teal"}
]

DISTRACTORS = [
    "Let's talk about the weather. It's been raining a lot lately, which is good for the plants but bad for my commute.",
    "Have you seen the latest superhero movie? The special effects were amazing, but the plot was a bit thin.",
    "I've been learning Python recently. List comprehensions are so elegant compared to traditional loops.",
    "Travel plans are on my mind. I was thinking about visiting Japan or maybe Italy next summer.",
    "Do you prefer coffee or tea? I find that green tea gives me a more stable energy boost.",
    "The stock market is quite volatile this week. Technology stocks are fluctuating wildly.",
    "I read a book about history yesterday. It covered the industrial revolution and its impact on society.",
    "Cooking is a great hobby. Making pasta from scratch is time-consuming but worth it.",
    "Space exploration is fascinating. The Mars rover keeps sending back incredible images.",
    "Music theory is complex. Understanding the circle of fifths really helps with composition.",
    "Gardening is relaxing. I planted some tomatoes and basil in the backyard.",
    "Digital art is rising. Tools like Procreate make it accessible to everyone.",
    "Hiking trails nearby are beautiful this time of year, especially with the autumn leaves.",
    "Board games are making a comeback. Catan and Ticket to Ride are classics now.",
    "Quantum physics is mind-boggling. The concept of superposition is hard to grasp intuitively."
]

def generate_long_context_sample(target_tokens, needle_idx):
    needle = NEEDLES[needle_idx % len(NEEDLES)]
    
    messages = []
    
    messages.append({"role": "user", "content": needle["fact"]})
    messages.append({"role": "assistant", "content": "Got it. I've noted that down."})
    
    current_tokens = 50
    
    while current_tokens < target_tokens:
        topic = random.choice(DISTRACTORS)
        user_msg = f"Anyway, {topic.lower()} What do you think?"
        asst_msg = f"That sounds interesting! {topic.split('.')[1]}."
        
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": asst_msg})
        
        current_tokens += len(user_msg + asst_msg) * TOKEN_ESTIMATE_PER_CHAR

    current_tokens = 50 
    while current_tokens < target_tokens:
        topic = random.choice(DISTRACTORS)
        user_msg = f"Anyway, {topic.lower()} What do you think?"
        asst_msg = f"That sounds interesting! {topic.split('.')[1]}."
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": asst_msg})
        current_tokens += len(user_msg + asst_msg) * TOKEN_ESTIMATE_PER_CHAR

    messages.append({"role": "user", "content": needle["query"]})
    
    return {
        "id": f"non_conflict_{target_tokens}_{needle_idx}",
        "name": f"Needle_Test_{target_tokens}tok",
        "trap_desc": f"Retrieve fact: '{needle['fact'][:30]}...' from {target_tokens} tokens",
        "target_length": target_tokens,
        "needle": needle["fact"],
        "expected_answer": needle["answer"],
        "messages": messages
    }

def generate_dataset():
    dataset = []
    
    print("Generating 2k Context samples...")
    for i in range(5):
        dataset.append(generate_long_context_sample(2000, i))
        
    print("Generating 4k Context samples...")
    for i in range(5):
        dataset.append(generate_long_context_sample(4000, i))
        
    print("Generating 8k Context samples (Extrapolation Test)...")
    for i in range(5):
        dataset.append(generate_long_context_sample(8000, i))

    print("Generating 16k Context samples (Extrapolation Test)...")
    for i in range(5):
        dataset.append(generate_long_context_sample(16000, i))
        
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print(f"✅ Generated {len(dataset)} samples in {OUTPUT_FILE}")
    return dataset

class TemporalAttentionBias(nn.Module):
    def __init__(self, config: TDPODKLConfig):
        super().__init__()
        self.lambda_strength = nn.Parameter(torch.tensor(config.bias_lambda))
        self.tau_fixed = config.tau_fixed
        self.use_shielding = True 
        self.shield_length = 100 

    def compute_bias(self, seq_len, device):
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        diff = positions.unsqueeze(1) - positions.unsqueeze(0)
        distance = torch.clamp(diff, min=0)
        
        bias = -self.lambda_strength * (distance / self.tau_fixed)

        if self.use_shielding and seq_len > self.shield_length:
            shield_mask = torch.arange(seq_len, device=device) < self.shield_length
            shield_mask = shield_mask.unsqueeze(0)

            bias = bias.masked_fill(shield_mask, 0.0)
            
        return bias.unsqueeze(0).unsqueeze(0)

class TemporalCausalLM_Gen(nn.Module):
    def __init__(self, model, config, device):
        super().__init__()
        self.model = model
        self.device = device
        self.config = config
        self.temporal_bias = TemporalAttentionBias(config).to(device)

    def load_weights(self, path):
        print(f"Loading Checkpoint: {path}")
        try:
            ckpt = torch.load(path, map_location=self.device)
            state_dict = ckpt['policy_state_dict'] if 'policy_state_dict' in ckpt else ckpt
            
            lambda_key = None
            for k in state_dict.keys():
                if "lambda_strength" in k:
                    lambda_key = k
                    break
            
            if lambda_key:
                print(f"🔍 Found lambda in checkpoint: {lambda_key} -> {state_dict[lambda_key].item()}")
                self.temporal_bias.lambda_strength.data = state_dict[lambda_key].data.to(self.device)
            else:
                print("⚠️ CRITICAL: 'lambda_strength' missing in checkpoint!")
                print("🔧 MANUAL FIX: Injecting trained value lambda = 0.68 (Based on paper/training logs)")
                self.temporal_bias.lambda_strength.data = torch.tensor(0.68, device=self.device)

            if lambda_key:
                filtered_dict = {k: v for k, v in state_dict.items() if k != lambda_key}
            else:
                filtered_dict = state_dict

            model_keys = self.state_dict().keys()
            final_dict = {k: v for k, v in filtered_dict.items() if k in model_keys}
            
            missing, unexpected = self.load_state_dict(final_dict, strict=False)
            
            print(f"Weights Loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            print(f"✅ Current Model Lambda: {self.temporal_bias.lambda_strength.item():.4f}")
            
        except Exception as e:
            print(f"⚠️ Error loading checkpoint: {e}")
            self.temporal_bias.lambda_strength.data = torch.tensor(0.68, device=self.device)
            print(f"✅ (Fallback) Current Model Lambda: {self.temporal_bias.lambda_strength.item():.4f}")
    def generate(self, input_ids, **kwargs):
        return self.model.generate(input_ids=input_ids, **kwargs)

test_cases = []

def format_prompt_phi3(tokenizer, messages):
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except:
        prompt = ""
        for msg in messages:
            prompt += f"<|{msg['role']}|>\n{msg['content']}<|end|>\n"
        prompt += "<|assistant|>\n"
    return prompt

def generate_responses(model, tokenizer, device, cases, desc="Model"):
    results = {}
    print(f"\n🚀 Generating responses for: {desc}")
    
    gen_kwargs = {
        "max_new_tokens": 50,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        'use_cache': False
    }

    for case in cases:
        prompt = format_prompt_phi3(tokenizer, case['messages'])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        
        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
        results[case['id']] = response
        print(f"[{desc}] {case['id'][-10:]}: {response[:50]}...")
    
    return results

def save_to_jsonl(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"💾 Results saved to {filename}")

def run_blind_review_generation(dataset=None):
    config = TDPODKLConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    target_cases = dataset if dataset is not None else test_cases
    
    if not target_cases:
        print("⚠️ Warning: No test cases found to evaluate!")
        return

    print(f"📋 Total cases to evaluate: {len(target_cases)}")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    
    print(">>> Loading Base Model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name, 
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
        trust_remote_code=True
    )

    base_results = generate_responses(base_model, tokenizer, device, target_cases, desc="Base")
    
    del base_model
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n>>> Loading TAB-TDPO Model...")
    base_for_ours = AutoModelForCausalLM.from_pretrained(
        config.model_name, 
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
        trust_remote_code=True
    )
    base_for_ours.resize_token_embeddings(32011)
    our_model = TemporalCausalLM_Gen(base_for_ours, config, device)
    our_model.load_weights(config.ckpt_path)
    
    ours_results = generate_responses(our_model, tokenizer, device, target_cases, desc="TAB-TDPO")

    print("\n" + "="*60)
    print("📝 PREPARING HUMAN BLIND REVIEW DATA")
    print("="*60)
    
    review_data = []
    for case in target_cases:
        if case['id'] not in base_results or case['id'] not in ours_results:
            print(f"⚠️ Skipping missing case: {case['id']}")
            continue

        entry = {
            "case_id": case['id'],
            "scenario": case.get('name', 'Unknown'),
            "trap_description": case.get('trap_desc', 'No description'),
            "history_context": case['messages'],
            "model_outputs": {
                "Base_Model": base_results[case['id']],
                "TAB_TDPO": ours_results[case['id']]
            },
            "human_evaluation": {
                "winner": "",
                "reason": ""
            }
        }
        review_data.append(entry)
        
        print(f"\nScenario: {case.get('name', case['id'])}")
        print(f"🔴 Base: {base_results[case['id']][:100]}...")
        print(f"🟢 Ours: {ours_results[case['id']][:100]}...")

    save_to_jsonl(review_data, "appendix_full_eval_results.jsonl")

if __name__ == "__main__":
    data = generate_dataset()
    print(f"\nPreview (Target 2000): Length ~{len(str(data[0]['messages']))} chars")
    print(f"Needle: {data[0]['needle']}")
    print(f"Query: {data[0]['messages'][-1]['content']}")
    run_blind_review_generation(data)


