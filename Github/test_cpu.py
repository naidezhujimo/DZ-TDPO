import torch
import sys
import os
import shutil
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

# 确保能导入 dz_tdpo
sys.path.append(os.getcwd())

from dz_tdpo.config import TDPODKLConfig
from dz_tdpo.model import TemporalCausalLM
from dz_tdpo.trainer import TDPODKLTrainer
from dz_tdpo.data.dataset import TemporalPreferenceDataset, TemporalPreferenceSample

def run_smoke_test():
    print("🚀 开始 CPU 冒烟测试 (Dry Run)...")
    
    # 1. 准备假环境 (修改版：Windows 安全逻辑)
    output_dir = "./tmp_test_output"
    
    # 不强制删除，而是如果存在就直接用，或者换个名字
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        # 如果文件夹已存在，尝试清空里面的文件，而不是删除文件夹本身
        # 这样避免了文件夹锁的问题
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"⚠️ 警告: 无法清理旧文件 {file_path}, 但我们将继续尝试运行。原因: {e}")

    device = torch.device("cpu") # 强制使用 CPU
    
    # 2. 使用微型模型 (gpt2) 代替 Phi-3，避免爆显存
    print("📦 初始化微型模型 (GPT-2)...")
    tiny_model_name = "gpt2" # 非常小，任何电脑都能跑
    tokenizer = AutoTokenizer.from_pretrained(tiny_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # 添加必要的 special tokens
    tokenizer.add_special_tokens({'additional_special_tokens': ["<|user|>", "<|assistant|>", "<|end|>"]})
    
    config_base = AutoConfig.from_pretrained(tiny_model_name)
    base_model = AutoModelForCausalLM.from_config(config_base) # 随机初始化，不下载权重，更快
    base_model.resize_token_embeddings(len(tokenizer))
    
    ref_base = AutoModelForCausalLM.from_config(config_base)
    ref_base.resize_token_embeddings(len(tokenizer))
    
    # 3. 初始化 DZ-TDPO 组件
    print("⚙️ 初始化 Config 和 Wrapper...")
    config = TDPODKLConfig(
        model_name=tiny_model_name,
        use_temporal_bias=True, # 测试我们的核心组件
        use_adaptive_tau=False, # 关掉 adaptive 以免需要下载 SBERT
        loss_type="tdpo",
        max_context_length=128
    )
    
    policy_model = TemporalCausalLM(base_model, config, device)
    ref_model = TemporalCausalLM(ref_base, config, device)
    
    # 4. 构造假数据 (Mock Data)
    print("📝 构造假数据...")
    dummy_samples = []
    for i in range(4): # 造4条数据
        ctx_len = 20
        seq_len = 10
        # 随机生成 token ID
        ctx_ids = torch.randint(0, len(tokenizer), (ctx_len,))
        # 模拟 dataset 输出结构
        dummy_samples.append(TemporalPreferenceSample(
            context_ids=ctx_ids,
            context_turns=[0, 10, 20],
            chosen_reply_ids=torch.randint(0, len(tokenizer), (seq_len,)),
            rejected_reply_ids=torch.randint(0, len(tokenizer), (seq_len,)),
            turn_id=2,
            total_turns=3
        ))
        
    train_dataset = TemporalPreferenceDataset(dummy_samples, tokenizer, config)
    
    # 5. 初始化 Trainer
    print("🏃 初始化 Trainer...")
    trainer = TDPODKLTrainer(
        policy_model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        config=config,
        train_dataset=train_dataset,
        val_dataset=None,
        device=device,
        output_dir=output_dir,
        gradient_accumulation_steps=1
    )
    
    # 6. 尝试运行 1 个 step
    print("🔥 尝试运行训练 Step...")
    try:
        # 取出一个 batch
        dataloader = trainer.train_loader
        batch = next(iter(dataloader))
        
        # 运行一步训练
        metrics = trainer.train_step(batch)
        print(f"✅ 训练步成功! Loss: {metrics['loss']:.4f}")
        
        # 尝试保存
        trainer.save_checkpoint("test_ckpt.pt")
        print("✅ 保存检查点成功!")
        
    except Exception as e:
        print(f"❌ 测试失败! 错误信息:\n{e}")
        import traceback
        traceback.print_exc()
        return

    # 清理
    shutil.rmtree(output_dir)
    print("\n✨ 恭喜！代码结构验证通过。可以发布到 GitHub！")

if __name__ == "__main__":
    run_smoke_test()