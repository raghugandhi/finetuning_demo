import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys

MODEL_ID = "google/gemma-3-270m-it"
ADAPTER_DIR = "gemma3-270m-email-lora-adapter"

def print_header(text):
    print("\n" + "="*50)
    print(f"🚀 {text.upper()}")
    print("="*50)

print_header("Initializing Interactive Inference")
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print(f"Loading base model to {device.upper()}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32, 
).to(device)

try:
    print(f"Loading (but NOT applying yet) your fine-tuned LoRA adapters from '{ADAPTER_DIR}'...")
    # Using adapter_name allows us to toggle it on and off dynamically
    model = PeftModel.from_pretrained(model, ADAPTER_DIR, adapter_name="professional")
except Exception as e:
    print(f"\n❌ ERROR: Could not find the trained adapter folder '{ADAPTER_DIR}'.")
    print("Please run 'python3 interactive_train.py' first to train the model!")
    sys.exit()

model.eval()

import re

def parse_natural_input(user_input):
    """
    Parses input like: 'give me an email in polite and confident for the "hey cany do this work asap"?'
    Returns: (prefix, tone_label, content)
    """
    input_lower = user_input.lower()
    
    # 1. Detect Tones
    found_tones = []
    if "polite" in input_lower: found_tones.append("polite")
    if "confident" in input_lower: found_tones.append("confident")
    if "professional" in input_lower: found_tones.append("professional")
    if "friendly" in input_lower: found_tones.append("friendly")
    
    # Handle the hybrid case first
    if "polite" in found_tones and "confident" in found_tones:
        prefix = "Rewrite polite and confident:"
        tone_label = "POLITE & CONFIDENT"
    elif found_tones:
        # Use the first tone found if not hybrid
        primary_tone = found_tones[0]
        prefix = f"Rewrite {primary_tone}:"
        tone_label = primary_tone.upper()
    else:
        # Default
        prefix = "Rewrite friendly:"
        tone_label = "FRIENDLY (Default)"

    # 2. Extract Content
    # Try to find text inside quotes first
    quotes_match = re.search(r'["\'](.*?)["\']', user_input)
    if quotes_match:
        content = quotes_match.group(1).strip()
    else:
        # If no quotes, take everything after the last colon or common keywords
        keywords = ["for the", "for", "rewrite:", "rewrite"]
        content = user_input
        for kw in keywords:
            if kw in input_lower:
                content = user_input[input_lower.rfind(kw) + len(kw):].strip()
                break
        
        # Clean up any trailing punctuation like '?'
        content = content.rstrip('?')

    return prefix, tone_label, content

print_header("Model Ready!")
print("Enter a command like: 'Rewrite polite and confident: [your message]'")
print("Or: 'Give me a professional version of \"I am leaving\"'")
print("Type 'quit' or 'exit' to stop.")

while True:
    raw_input = input("\n[COMMAND]: ").strip()
    
    if raw_input.lower() in ["quit", "exit"]:
        print("Goodbye!")
        break
    
    if not raw_input:
        continue
    
    prefix, tone_label, content = parse_natural_input(raw_input)
    
    if not content or content == raw_input:
        print(f"⚠️ Could not clearly separate tone from content. Using full input as content.")
        content = raw_input

    print(f"✨ Detected Tone: {tone_label}")
    print(f"📝 Extracted Content: '{content}'")

    # Format the prompt using the Gemma Chat Template
    prompt = f"{prefix} {content}"
    messages = [{"role": "user", "content": prompt}]
    
    # Pre-process
    inputs = tokenizer.apply_chat_template(
        messages, 
        tokenize=True, 
        add_generation_prompt=True, 
        return_tensors="pt"
    ).to(device)

    # Generate - BASE MODEL (Generic)
    print(f"\nGenerating Base (Generic) Response...")
    with torch.no_grad():
        with model.disable_adapter():
            out_base = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
            )

    # Decode - BASE MODEL
    input_length = inputs["input_ids"].shape[1]
    base_tokens = out_base[0][input_length:]
    base_text = tokenizer.decode(base_tokens, skip_special_tokens=True).strip()

    # Generate - LORA MODEL (Fine-tuned)
    print(f"Generating Fine-tuned ({tone_label}) Response...")
    model.set_adapter("professional")
    with torch.no_grad():
        out_lora = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
        )

    # Decode - LORA MODEL
    lora_tokens = out_lora[0][input_length:]
    lora_text = tokenizer.decode(lora_tokens, skip_special_tokens=True).strip()

    print("\n" + "*" * 50)
    print(f"[BASE MODEL]:\n{base_text if base_text else '(Empty output)'}")
    print("-" * 50)
    print(f"[FINE-TUNED {tone_label}]:\n{lora_text if lora_text else '(Empty output)'}")
    print("*" * 50)
