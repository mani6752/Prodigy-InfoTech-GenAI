# download_data.py
# Downloads the Shakespeare dataset for fine-tuning GPT-2

import urllib.request
import os

DATA_DIR = "data"
FILE_NAME = "shakespeare.txt"
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

os.makedirs(DATA_DIR, exist_ok=True)
output_path = os.path.join(DATA_DIR, FILE_NAME)

print("Downloading Shakespeare dataset...")
urllib.request.urlretrieve(URL, output_path)

# Verify
with open(output_path, "r", encoding="utf-8") as f:
    content = f.read()

print(f"Downloaded {len(content)} characters ({len(content.split())} words)")
print(f"Saved to: {output_path}")
print("\nSample preview:")
print(content[:500])
