import os
# os.environ["BNB_CUDA_VERSION"] = "117"
os.environ["HF_HOME"] = "/home/sayantan/Alzheimer/LLM/unsloth-models/hf_cache" # temporary set the cache dir


import unsloth
from unsloth import FastLanguageModel
import torch


MODEL_ID = "sayantanBiswas/mistral-7b-v0.3"  
MAX_SEQ_LENGTH = 4096
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_ID,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = DTYPE,
    load_in_4bit = load_in_4bit,
    cache_dir='/home/sayantan/Alzheimer/LLM/unsloth-models'
)


# api-keys initialiser
from dotenv import load_dotenv
hf_token = os.environ['HF_TOKEN']

# Save to q4_k_m GGUF
model.save_pretrained_gguf("mistral-7b-v0.3-q4_k_m", tokenizer, quantization_method = "q4_k_m")
model.push_to_hub_gguf("sayantanBiswas/mistral-7b-v0.3-q4_k_m", tokenizer, quantization_method = "q4_k_m", token = hf_token)

