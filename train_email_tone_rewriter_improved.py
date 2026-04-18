import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# 1) Configuration
MODEL_ID = "google/gemma-3-270m-it"
ADAPTER_PATH = "gemma3-270m-email-lora-adapter-improved"
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# 2) Load Model & Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float32,
).to(device)

# 3) LoRA Config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 4) Load & Format Dataset
ds = load_dataset("json", data_files="emails.jsonl", split="train")

def format_prompts(ex):
    messages = [
        {"role": "user", "content": ex["instruction"]},
        {"role": "assistant", "content": ex["output"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

ds = ds.map(format_prompts)
ds = ds.train_test_split(test_size=0.1, seed=42)
train_ds, eval_ds = ds["train"], ds["test"]

# 5) Improved Training Config
args = SFTConfig(
    output_dir="gemma3-270m-email-lora-improved",
    per_device_train_batch_size=1, # Reduced from 4
    gradient_accumulation_steps=8, # Increased from 2
    dataloader_pin_memory=False,   # Fix for MPS crash
    learning_rate=5e-5, # Stable LR
    num_train_epochs=15, # Deep learning
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    logging_steps=1,
    eval_strategy="steps",
    eval_steps=10,
    save_steps=50,
    save_total_limit=1,
    report_to="none",
    max_length=512,
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    args=args,
)

# 6) Train and Save
print(f"🚀 Starting Improved Training on {device.upper()}...")
trainer.train()
trainer.model.save_pretrained(ADAPTER_PATH)
tokenizer.save_pretrained(ADAPTER_PATH)
print(f"✅ Training complete! Improved adapter saved to {ADAPTER_PATH}")
