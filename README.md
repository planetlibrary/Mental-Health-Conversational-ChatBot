# 🧠 Mental Health Conversational ChatBot — GGUF Conversion Pipeline

This repository provides a pipeline to fine-tune a language model on mental health dialogue data using [Unsloth](https://github.com/unslothai/unsloth) and then convert it to [GGUF](https://github.com/ggerganov/llama.cpp/blob/master/docs/gguf.md) format for use with [`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python).

---

## 📌 Overview

1. **Fine-tune model** using `unsloth/mistral-7b-v0.3-bnb-4bit`.
2. **Upload** the fine-tuned model to Hugging Face Hub.
3. **Convert** the uploaded model to `GGUF` (quantized `q4_k_m`) using `lora_to_gguf.py`.
4. Use the model with `llama-cpp-python`.

---

## 🔧 Setup

### 1. Clone & Build `llama.cpp`

Before running the GGUF conversion script, you **must** set up the `llama.cpp` repository to build the `llama-quantize` binary. Run the following:

```bash
chmod +x setup_llama_cpp.sh # to make it executable
```
Then run the script to setup the llama-cpp

```bash
./setup_llama_cpp.sh
```