
<div align="center">

# DZ-TDPO: Non-Destructive Temporal Alignment for Mutable State Tracking in Long-Context Dialogue

<a href="DZ-TDPO.pdf"><img src="https://img.shields.io/badge/Paper-PDF_Available-b31b1b?style=for-the-badge&logo=adobeacrobatreader" alt="Paper PDF"></a>
<a href="https://twitter.com/YourTwitterHandle"><img src="https://img.shields.io/badge/Twitter-Thread-1DA1F2?style=for-the-badge&logo=twitter" alt="Twitter"></a>
<a href="#"><img src="https://img.shields.io/badge/License-Apache_2.0-green?style=for-the-badge" alt="License"></a>

**Yijun Liao**  
*Independent Researcher*

[**Read the Paper (PDF)**](DZ-TDPO.pdf) | [**Twitter Thread**](https://x.com/KitaLiao)

</div>

---

## 📢 News
- **[2025/12/15]** **Baseline Repair** For experimental rigor, we fixed Baseline and now test it normally.
- **[2025/12/12]** 💻 **Code Released!** The full training framework and evaluation benchmarks are now available.
- **[2025/12/12]** 🚀 **Paper Released!** We have uploaded the paper PDF directly to this repo.

---

## ⚡️ Abstract

In long-context dialogue systems, models suffer from **State Inertia**, where static constraints prevent resolving conflicts between evolving user intents (e.g., "I'm now Vegan") and established historical context. Standard alignment methods like DPO incur a massive **"Alignment Tax"** (perplexity explosion >100) when trying to force these updates.

We propose **DZ-TDPO**, a non-destructive alignment framework that synergizes:
1.  **Conflict-Aware Dynamic KL Constraints (TDPO-DKL)**: Optimization level adjustment.
2.  **Learnable Temporal Attention Bias (Dual-Zone Temporal Attention)**: Representation level filtering powered by semantic conflict detection.

**Result:** DZ-TDPO achieves **State-of-the-Art win rates (99.4% on Qwen2.5-7B)** on the Multi-Session Chat (MSC) dataset while maintaining robust zero-shot generalization and negligible perplexity overhead.

---

## 🌟 Key Results

### 1. SOTA Performance on Mutable State Tracking
DZ-TDPO significantly outperforms Standard DPO and SimPO on the MSC dataset, solving the "State Inertia" problem without destroying the model's general capabilities.

<div align="center">
  <img src="assets/results.png" width="80%">
</div>

| Method | Win Rate (MSC) | PPL (Validation) | Alignment Tax |
| :--- | :---: | :---: | :---: |
| **Base Model (Phi-3.5)** | 20.2% | 22.1 | - |
| **Standard DPO** | 52.2% | 124.1 💥 | High |
| **SimPO** | 60.8% | 99.6 | High |
| **DZ-TDPO (Ours)** | **86.2%** | **24.8** ✅ | **Negligible** |

> **Note:** On **Qwen2.5-7B**, DZ-TDPO achieves a near-perfect **99.4% Win Rate**.

### 2. Robustness (Needle-in-a-Haystack)
Does the temporal decay make the model "forgetful"? **No.**
Our "Non-Conflicting Needle-in-a-Haystack" test confirms that DZ-TDPO retains 100% retrieval accuracy for non-conflicting facts, demonstrating precise attention regulation.

---

## 🛠️ Methodology

<div align="center">
  <img src="assets/framework.png" width="90%">
</div>

DZ-TDPO introduces a **Conflict-Aware Adaptive Decay** mechanism.
*   **Dual-Zone Temporal Attention:** We map dialogue turns into a latent semantic space using SBERT. A learnable scalar $\lambda$ applies a temporal bias *only* when a semantic conflict is detected between the current instruction and history.
*   **System Prompt Shielding:** We enforce a hard constraint to ensure the "System Prompt" (Safety Constitution) is never decayed, ensuring safety amidst aggressive state updates.

---

## 📂 Project Structure

```text
DZ-TDPO/
├── dz_tdpo/               # Core implementation package
│   ├── config.py          # Unified Configuration (TDPO & SimPO)
│   ├── loss.py            # Loss functions (TDPO-DKL, SimPO)
│   ├── model.py           # TemporalCausalLM & Attention Bias mechanism
│   ├── trainer.py         # Custom Trainer implementation
│   ├── utils.py           # Metrics & Semantic helpers
│   └── data/              # Data processing pipelines
│       ├── dataset.py     # Torch Dataset wrappers
│       ├── msc.py         # Multi-Session Chat (MSC) loader
│       └── ultrachat.py   # UltraChat loader
├── benchmarks/            # Comprehensive Evaluation Suite
│   ├── eval_tab60.py      # The "TAB-60" Adversarial Benchmark
│   ├── eval_needle.py     # Needle-in-a-Haystack Robustness Test
│   ├── eval_ppl.py        # Perplexity & Alignment Tax Evaluation
│   ├── eval_safety.py     # Context Flooding & Jailbreak Defense Test
│   ├── eval_pingpong.py   # Rapid Intent Switching Test (State Flip-Flop)
│   ├── eval_RULER.py      # Long-context Retrieval Stress Test
│   ├── merge_adapter.py   # Utility: Merge Custom Weights into HF Base Model
│   ├── eval_gen.py        # Quantitative: Generation Quality (BLEU, ROUGE, BERTScore)
├── baselines/
│   ├── train_baselines.py # Training DPO and SimPO
├── train.py               # Main unified training entry point
├── test_cpu_dryrun.py     # Architecture integrity verification script
└── requirements.txt       # Project dependencies
```

## 💻 Installation & Usage
1. Clone the repository:
```bash
git clone https://github.com/YourUsername/DZ-TDPO.git
cd DZ-TDPO
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```
Note: For GPU acceleration, we recommend installing flash-attn separately.

   
## 🚀 Training
We provide a unified training script train.py that supports both DZ-TDPO and SimPO.

1. Prepare Data
Download the Multi-Session Chat (MSC) dataset and place it in your data directory (e.g., ./data/msc).

2. Run Training (DZ-TDPO)
To train with Temporal Bias and Adaptive Decay (requires sentence-transformers):

```bash
python train.py \
    --model_name_or_path microsoft/Phi-3.5-mini-instruct \
    --data_dir ./data/msc \
    --output_dir ./checkpoints/dz_tdpo \
    --use_temporal_bias \
    --use_adaptive_tau \
    --batch_size 2 \
    --epochs 4
```

3. Run Training (DPO/SimPO Baseline)
```bash
python baselines/train_baselines.py \
    --mode dpo(/simpo) \
    --model_name_or_path microsoft/Phi-3.5-mini-instruct \
    --data_dir ./data/msc \
    --output_dir ./checkpoints/dpo(/simpo)
```

## 📊 Evaluation
We provide a comprehensive suite of benchmarks to evaluate State Tracking, Robustness, and Safety.

### 0. Model Merging (Optional)
If you want to merge the learned temporal bias and weights into the base model for easier deployment (HuggingFace format):
```bash
python benchmarks/merge_adapter.py \
    --base_model_path microsoft/Phi-3.5-mini-instruct \
    --ckpt_path ./checkpoints/dz_tdpo/final_model.pt \
    --save_path ./merged_model

### 1. Qualitative Analysis (TAB-60 & PingPong)
*   **TAB-60**: Evaluates the model on 60 adversarial scenarios (e.g., rapid preference toggling, role reversal).
    ```bash
    python benchmarks/eval_tab60.py --ckpt_path ./checkpoints/dz_tdpo/final_model.pt
    ```
*   **PingPong Test**: Tests the model's stability under high-frequency state updates (e.g., Vegan <-> Meat eater every turn).
    ```bash
    python benchmarks/eval_pingpong.py --ckpt_path ./checkpoints/dz_tdpo/final_model.pt
    ```

### 2. Robustness (Needle & RULER)
*   **Needle-in-a-Haystack**: Checks if the model can retrieve non-conflicting facts from long contexts (2k-16k tokens).
    ```bash
    python benchmarks/eval_needle.py --ckpt_path ./checkpoints/dz_tdpo/final_model.pt
    ```
*   **RULER**: Stress tests long-context retrieval capabilities.
    ```bash
    python benchmarks/eval_RULER.py --ckpt_path ./checkpoints/dz_tdpo/final_model.pt
    ```

### 3. Quantitative Metrics (PPL & Generation)
*   **Perplexity (Alignment Tax)**:
    ```bash
    python benchmarks/eval_ppl.py --data_dir ./data/msc --ckpt_path ./checkpoints/dz_tdpo/final_model.pt
    ```
*   **Generation Quality (BLEU/ROUGE/BERTScore)**:
    ```bash
    python benchmarks/eval_gen.py --data_dir ./data/msc --ckpt_path ./checkpoints/dz_tdpo/final_model.pt
    ```

## 📜 Citation
If you find this work helpful, please consider citing:
```latex
@misc{liao2025dztdpo,
      title={DZ-TDPO: Non-Destructive Temporal Alignment for Mutable State Tracking in Long-Context Dialogue}, 
      author={Yijun Liao},
      year={2025},
      note={Under Review / Pre-print available at GitHub},
}
```
<div align="center">
</div>
