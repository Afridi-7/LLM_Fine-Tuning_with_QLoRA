# 🔧 LLM Fine-Tuning with QLoRA
### Fine-tune any LLM on a free GPU — clean, minimal, and actually works.


## 🧠 What is this?

A clean, well-commented Jupyter notebook that walks you through **fine-tuning a Large Language Model from scratch** using **QLoRA** — the technique that makes it possible to train billion-parameter models on consumer GPUs and even free cloud instances.

No bloated code. No unnecessary complexity. Just the essentials, explained clearly.

Built as a **practice project** to understand fine-tuning internals before applying it to domain-specific research (medical AI). If you're trying to learn the same, this is your starting point.

---

## ⚡ Quick Concepts

**Fine-Tuning** — A pre-trained LLM already knows a lot about the world. Fine-tuning redirects that knowledge toward your specific task. You're not building a brain — you're giving it a job description.

**PEFT** — Parameter Efficient Fine-Tuning. Instead of updating all billions of parameters, freeze 99% of the model and only train a tiny slice. Full fine-tuning = renovating the entire house. PEFT = repainting one room. Same result, 100x cheaper.

**LoRA** — Low-Rank Adaptation. Doesn't touch the original weights at all. Sneaks in small trainable matrices alongside frozen layers. Only ~1% of parameters are trained. The model still learns. Wild, right?

**QLoRA** — LoRA but make it run anywhere. Compresses the model to 4-bit precision before applying LoRA. A model that needs 10GB of VRAM suddenly fits in 3GB.

---

## 📊 My Results

| Setting | Value |
|---|---|
| **Base model** | Microsoft Phi-2 (2.7B parameters) |
| **Method** | QLoRA — 4-bit NF4 quantization |
| **Dataset** | Alpaca instruction dataset |
| **Samples** | 500 |
| **Parameters trained** | ~1% of total |
| **Eval loss** | 0.334 |
| **Perplexity** | 1.40 ✅ |
| **GPU** | Free Google Colab T4 |

> Perplexity under 2.0 is considered outstanding for this setup.

---

## 📁 Notebook Structure

```
1. Install Dependencies
2. Imports & Configuration        ← change model/dataset here, one place
3. Load Model (4-bit quantized)
4. Attach LoRA Adapters
5. Prepare & Format Dataset
6. Train
7. Evaluate (loss + perplexity)
8. Save Adapter Weights
9. Inference & Testing
10. Reload Model in New Session
```

---

## 🚀 Getting Started

### Option 1 — Google Colab (Recommended)
Click the badge below to open directly in Colab with a free T4 GPU:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/repo-name/blob/main/llm_finetuning.ipynb)

### Option 2 — Run Locally
```bash
git clone https://github.com/yourusername/repo-name.git
cd repo-name
pip install -r requirements.txt
jupyter notebook llm_finetuning.ipynb
```

---

## 🛠️ Requirements

```
transformers
datasets
peft
trl
bitsandbytes
accelerate
sentencepiece
protobuf
torch >= 2.0
```

Or just run the first cell in the notebook — it installs everything automatically.

---

## 🔄 Swap the Model

Change one line in the config cell:

```python
# Lightweight — free Colab T4
MODEL_NAME = "microsoft/phi-2"

# Stronger — still fits on free T4
MODEL_NAME = "mistralai/Mistral-7B-v0.1"

# Best open source — free T4 with 4-bit
MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
```

Everything else stays the same.

---

## 🔄 Swap the Dataset

Replace `DATASET_NAME` and update the `format_prompt()` function to match your data format. The notebook uses Alpaca out of the box but works with any HuggingFace dataset or custom data.

```python
DATASET_NAME = "tatsu-lab/alpaca"       # default
DATASET_NAME = "your-username/your-dataset"  # custom
```

---

## ⚙️ Key Hyperparameters

| Parameter | Default | When to change |
|---|---|---|
| `LORA_R` | 16 | Increase for harder tasks (32, 64) |
| `LORA_ALPHA` | 32 | Keep at 2× LORA_R |
| `LEARNING_RATE` | 2e-4 | Lower if training is unstable |
| `MAX_SEQ_LEN` | 256 | Lower to 256 for faster runs |
| `NUM_SAMPLES` | 500 | Set to `None` for full dataset |
| `EPOCHS` | 3 | Increase to 5–10 for small datasets |

---

## 💡 Tips

- **First time?** Set `NUM_SAMPLES = 100` and `EPOCHS = 1` for a 10-minute smoke test before committing to a full run.
- **Slow training?** Drop `MAX_SEQ_LEN` to 256 — cuts time roughly in half.
- **Watch eval loss**, not just train loss. If eval loss starts rising while train loss drops, you're overfitting — stop early.
- **`load_best_model_at_end=True`** is already set — so even if later epochs overfit, you get the best checkpoint automatically.

---

## 🗺️ What's Next

This notebook is a foundation. Here's where it can go:

- [ ] Fine-tune on a **medical domain dataset** for clinical NLP tasks
- [ ] Add **before/after comparison** — base model vs fine-tuned side by side
- [ ] Experiment with **Mistral-7B or LLaMA-3** for stronger baselines
- [ ] Push the fine-tuned adapter to **HuggingFace Hub**
- [ ] Add **W&B integration** for experiment tracking

---

## 🤝 Contributing

Found a bug? Have an improvement? PRs are welcome.

1. Fork the repo
2. Create your branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push and open a PR

---

## 📄 License

MIT — use it, modify it, build on it. Just give credit if it helped.

---

## ⭐ Support

If this saved you hours of debugging or helped you understand fine-tuning — a **star on this repo would genuinely mean a lot**. It helps others find it too.
