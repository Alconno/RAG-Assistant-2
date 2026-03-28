from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Paths
base_model_path = "Qwen/Qwen3-0.6B"
lora_checkpoint = "./models/qwen06b/instruct/checkpoint-57"

# Load base tokenizer/model
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(base_model_path, trust_remote_code=True, device_map="auto")

# Load LoRA weights
model = PeftModel.from_pretrained(base_model, lora_checkpoint)

# Inference
model.eval()
prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" \
         "<|im_start|>user\n" \
         "What is photosynthesis?\n" \
         "<|im_end|>\n<|im_start|>assistant\n"


inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
