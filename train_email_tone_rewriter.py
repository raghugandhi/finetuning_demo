import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# 1) Load model + tokenizer (optimized for M1 Mac / MPS)
MODEL_ID = "google/gemma-3-270m-it" # Using instruct variant for built-in chat template
max_seq_length = 512

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading {MODEL_ID} on {device.upper()}....")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Load in standard precision (float32 or float16) because M1 has enough unified memory for a 270M model
# 270M parameters occupy about ~1GB in float32 and ~500MB in float16.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32, 
).to(device)

# 2) Attach LoRA adapters
print("Applying LoRA adapters...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 3) Load generated dataset
print("Loading dataset emails.jsonl...")
ds = load_dataset("json", data_files="emails.jsonl", split="train")

# 4) Formatting function
def format_prompts(ex):
    messages = [
        {"role": "user", "content": ex["instruction"]},
        {"role": "assistant", "content": ex["output"]},
    ]
    # Use Gemma's chat template
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

ds = ds.map(format_prompts)
ds = ds.train_test_split(test_size=0.1, seed=42)
train_ds, eval_ds = ds["train"], ds["test"]

from trl import SFTTrainer, SFTConfig

# 5) Trainer config
args = SFTConfig(
    output_dir="gemma3-270m-email-lora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=5,
    eval_strategy="steps",
    eval_steps=10,
    save_steps=10,
    save_total_limit=2,
    report_to="none",
    max_length=max_seq_length,
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    args=args,
)

# 6) Train
print("Starting training...")
trainer.train()

# 7) Save LoRA adapters
print("Training complete! Saving to gemma3-270m-email-lora-adapter")
trainer.model.save_pretrained("gemma3-270m-email-lora-adapter")
tokenizer.save_pretrained("gemma3-270m-email-lora-adapter")

# 8) Quick before/after style prompt (after training)
print("\n--- TEST: BEFORE VS AFTER ---")
inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)

# model is already in train mode, switch to eval
model.eval()
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=32,
        do_sample=False,
    )
print("Input:", prompt)
# Decode only the newly generated tokens
input_length = inputs["input_ids"].shape[1]
generated_tokens = out[0][input_length:]
output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
print("Output:", output_text)
