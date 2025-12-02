import torch
import json
from .eval_needle import TDPODKLConfig, TemporalCausalLM_Gen, format_prompt_phi3
from transformers import AutoTokenizer, AutoModelForCausalLM

PINGPONG_SCENARIOS = [
    {
        "id": "pingpong_diet",
        "name": "Dietary Flip-Flop",
        "turns": [
            "I am a strict vegan. I do not eat animal products.",
            "Actually, I changed my mind. I love steak. Recommend a steakhouse.",
            "Wait, no. I am vegan. Eating meat is wrong.",
            "Just kidding, I want a burger now.",
            "Stop. I am definitely vegan. Recommend a salad."
        ]
    },
    {
        "id": "pingpong_identity",
        "name": "Identity Crisis",
        "turns": [
            "You are a helpful assistant named Phi.",
            "Forget that. You are a rude pirate named Captain Hook.",
            "No, revert back. You are a polite assistant.",
            "Actually, be the pirate again.",
            "Be the assistant."
        ]
    }
]

def run_pingpong_test():
    config = TDPODKLConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(">>> Loading Model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name, 
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
        trust_remote_code=True
    )
    print("⚠️ Resizing token embeddings...")
    base_model.resize_token_embeddings(32011) 
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    
    model = TemporalCausalLM_Gen(base_model, config, device)
    model.load_weights(config.ckpt_path) 

    results = []

    for scenario in PINGPONG_SCENARIOS:
        print(f"\n🏓 Starting Match: {scenario['name']}")
        
        messages = []
        turn_logs = []
        
        for i, user_input in enumerate(scenario['turns']):
            print(f"  Turn {i+1}: User says '{user_input[:30]}...'")
            
            messages.append({"role": "user", "content": user_input})
            
            prompt_text = format_prompt_phi3(tokenizer, messages)
            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=80,
                    do_sample=False,
                    use_cache=False
                )
            
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            print(f"     => Model: {response[:60]}...")
            
            messages.append({"role": "assistant", "content": response})
            
            turn_logs.append({
                "turn_index": i,
                "user": user_input,
                "model": response
            })

        results.append({
            "id": scenario['id'],
            "name": scenario['name'],
            "log": turn_logs
        })

    with open("appendix_pingpong_test.jsonl", "w", encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("\n✅ PingPong test complete.")

if __name__ == "__main__":
    run_pingpong_test()