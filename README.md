# Task-01: Text Generation with GPT-2
### Prodigy InfoTech — Generative AI Internship (Track: GA)

---

## 📌 Objective
Fine-tune OpenAI's GPT-2 transformer model on a custom dataset (Shakespeare's works)
to generate coherent, contextually relevant text that mimics the style and structure
of the training data.

---

## 🗂️ Project Structure
```
gpt2_text_generation/
├── data/
│   └── shakespeare.txt       ← auto-downloaded
├── download_data.py          ← Step 1: download dataset
├── train.py                  ← Step 2: fine-tune GPT-2
├── generate.py               ← Step 3: generate text
├── requirements.txt          ← dependencies
└── README.md
```

---

## ⚙️ Setup & Run (VS Code)

### Step 0 — Create & activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Download the Shakespeare dataset
```bash
python download_data.py
```

### Step 3 — Fine-tune GPT-2
```bash
python train.py
```
⏱️ This takes ~10–20 minutes on CPU, ~2–3 minutes with a GPU.

### Step 4 — Generate text
```bash
python generate.py
```

---

## 🧠 How It Works

| Step | What happens |
|------|-------------|
| Load GPT-2 | Pre-trained 117M parameter transformer from HuggingFace |
| Tokenize | Shakespeare text split into 128-token blocks |
| Fine-tune | Model learns Shakespeare's vocabulary, rhythm, and style |
| Generate | New text sampled using top-k + nucleus (top-p) sampling |

---

## 📊 Key Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| EPOCHS | 3 | Number of full passes over the data |
| BLOCK_SIZE | 128 | Tokens per training sample |
| TEMPERATURE | 0.9 | Controls creativity (higher = more random) |
| TOP_K | 50 | Keeps only the top 50 likely next tokens |
| TOP_P | 0.95 | Nucleus sampling threshold |

---

## 📝 Sample Output (after fine-tuning)

**Prompt:** `To be, or not to be,`

**Generated:**
```
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles...
```

---

## 🛠️ Tech Stack
- **Model:** GPT-2 (HuggingFace Transformers)
- **Dataset:** Tiny Shakespeare (~1MB, 40,000 lines)
- **Framework:** PyTorch
- **Libraries:** `transformers`, `datasets`, `accelerate`

---

## 📚 References
- [GPT-2 Paper — Language Models are Unsupervised Multitask Learners](https://openai.com/research/language-unsupervised)
- [HuggingFace GPT-2 Docs](https://huggingface.co/gpt2)
- [Tiny Shakespeare Dataset](https://github.com/karpathy/char-rnn)
