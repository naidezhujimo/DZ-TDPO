# DZ-TDPO

Implementation of **DZ-TDPO: Non-Destructive Temporal Alignment for Mutable State Tracking**.

## Installation

```bash
pip install -r requirements.txt
```

## Training
```base
python train.py \
    --model_name_or_path microsoft/Phi-3.5-mini-instruct \
    --data_dir ./data/msc \
    --use_temporal_bias
```

## Evaluation
```base
python benchmarks/eval_tab60.py --ckpt_path ./checkpoints/tdpo_final.pt
```