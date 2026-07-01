# train.py
import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from torch.utils.data import Dataset

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_NAME    = "gpt2"
TRAIN_FILE    = "data/shakespeare.txt"
OUTPUT_DIR    = "model_output"
BLOCK_SIZE    = 128
EPOCHS        = 3
BATCH_SIZE    = 4

# ─────────────────────────────────────────────
# CUSTOM DATASET CLASS (replaces TextDataset)
# ─────────────────────────────────────────────
class ShakespeareDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        tokens = tokenizer.encode(text)
        self.examples = []
        for i in range(0, len(tokens) - block_size, block_size):
            self.examples.append(torch.tensor(tokens[i:i + block_size]))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {"input_ids": self.examples[idx], "labels": self.examples[idx]}

# ─────────────────────────────────────────────
# LOAD TOKENIZER & MODEL
# ─────────────────────────────────────────────
print("Loading GPT-2 tokenizer and model...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
print(f"Model loaded — {model.num_parameters():,} parameters")

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
print("Preparing dataset...")
dataset = ShakespeareDataset(TRAIN_FILE, tokenizer, BLOCK_SIZE)
print(f"Dataset ready — {len(dataset)} samples")

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    save_steps=500,
    logging_steps=100,
    fp16=torch.cuda.is_available(),
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

print("\nStarting fine-tuning...")
trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nFine-tuned model saved to: {OUTPUT_DIR}/")