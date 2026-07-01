# generate.py
# Loads the fine-tuned GPT-2 model and generates Shakespeare-style text

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_DIR   = "model_output"    # path to your fine-tuned model
MAX_LENGTH  = 300               # max tokens to generate
TEMPERATURE = 0.9               # creativity: higher = more random
TOP_K       = 50                # top-k sampling
TOP_P       = 0.95              # nucleus sampling

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
print("Loading fine-tuned GPT-2 model...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
model     = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Running on: {device.upper()}")

# ─────────────────────────────────────────────
# GENERATE TEXT
# ─────────────────────────────────────────────
def generate_text(prompt: str) -> str:
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            inputs,
            max_length=MAX_LENGTH,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)


# ─────────────────────────────────────────────
# DEMO PROMPTS
# ─────────────────────────────────────────────
prompts = [
    "To be, or not to be,",
    "Shall I compare thee to a summer's day?",
    "What light through yonder window breaks?",
]

print("\n" + "="*60)
print("GPT-2 Fine-tuned on Shakespeare — Generated Text")
print("="*60)

for prompt in prompts:
    print(f"\n📝 Prompt: {prompt}")
    print("-"*50)
    result = generate_text(prompt)
    print(result)
    print("="*60)

# ─────────────────────────────────────────────
# INTERACTIVE MODE
# ─────────────────────────────────────────────
print("\n🎭 Enter your own prompt (or 'quit' to exit):")
while True:
    user_prompt = input("\nYour prompt: ").strip()
    if user_prompt.lower() in ("quit", "exit", "q"):
        break
    if user_prompt:
        print("\n" + generate_text(user_prompt))
