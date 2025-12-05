import torch
import numpy as np

_SEMANTIC_MODEL = None

def get_semantic_model(device='cpu'):
    global _SEMANTIC_MODEL
    if _SEMANTIC_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            print("🔄 Loading Semantic Embedding Model (all-MiniLM-L6-v2)...")
            _SEMANTIC_MODEL = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        except ImportError:
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")
    return _SEMANTIC_MODEL

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

def split_subset_metrics(metrics_list, mask):
    subset = [m for m, keep in zip(metrics_list, mask) if keep]
    if not subset:
        return {}
    
    keys = subset[0].keys()
    avg_metrics = {}
    for k in keys:
        values = [item[k] for item in subset if k in item]
        if values and isinstance(values[0], (int, float)):
            avg_metrics[k] = np.mean(values)
    
    return avg_metrics
