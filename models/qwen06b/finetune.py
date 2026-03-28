from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from torch.optim.lr_scheduler import OneCycleLR
from transformers import TrainingArguments, Trainer
import wandb
import math
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from datasets import load_dataset, concatenate_datasets
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="torch.utils.checkpoint")
warnings.filterwarnings("ignore", message="MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization")

# Model
print("Loading Model & Tokenizer")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    quantization_config=BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    ),
    device_map="auto",
    trust_remote_code=True,
)

print("Special tokens map:")
for key, val in tokenizer.special_tokens_map.items():
    print(f"{key}: {repr(val)}")

# Get dataset
print("Loading dataset")
alpaca_dataset = load_dataset("tatsu-lab/alpaca")['train'] # no context dataset
alpaca_dataset = alpaca_dataset.train_test_split(train_size=.01, seed=42)['train']
squadv2_dataset = load_dataset("rajpurkar/squad_v2", download_mode="force_redownload")['train'] # dataset with context
squadv2_dataset = squadv2_dataset.train_test_split(train_size=.01, seed=42)['train']

# Create prompts
print("Creating prompts")
def format_prompt(example):
    system_prompt = "You are a helpful assistant."
    context_str = f"Context:\n{example['context']}\n" if 'context' in example and example['context'] else ""
    question = f"{example['instruction']} - {example['input']}" if 'instruction' in example and example['instruction'] \
               else example.get('question', "")
    question_str = f"Question: {question}\n"
    answer = example['output'] if 'output' in example else \
             example['answers']['text'][0] if 'answers' in example and example['answers']['text'] else "I don't know"
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{context_str}"
        f"{question_str}"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"{answer}"
        f"<|im_end|>"
    )
    

alpaca_dataset = alpaca_dataset.map(lambda x: {"prompt": format_prompt(x)})
squadv2_dataset = squadv2_dataset.map(lambda x: {"prompt": format_prompt(x)})
dataset = concatenate_datasets([alpaca_dataset, squadv2_dataset])
dataset = dataset.shuffle(seed=42)

# Tokenize
print("Tokenizing dataset")
def tokenize_fn(batch):
    result = tokenizer(
        batch["prompt"],
        truncation=True,
        max_length=512,
        padding="longest",
    )
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_dataset = dataset.map(tokenize_fn, batched=True)

# Setup LoRA fine-tuning
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

# Training
wandb.init(project="qwen3-instruct")

epochs      = 3
n_devices   = 1
n_examples    = len(dataset)
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
steps_per_epoch     = math.ceil(n_examples/(per_device_train_batch_size*gradient_accumulation_steps*n_devices))
total_steps         = steps_per_epoch * epochs
optimizer   = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
scheduler   = OneCycleLR(
    optimizer,
    max_lr=1e-3,
    total_steps=total_steps,
    pct_start=0.3,
    anneal_strategy="cos",
    div_factor=25.0,
    final_div_factor=50.0,
)

training_args = TrainingArguments(
    output_dir="./models/qwen06b/instruct",
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=False,
    logging_dir="./wandb",
    logging_strategy="steps",
    logging_steps=1,
    num_train_epochs=epochs,
    fp16=True,
    save_strategy="epoch",
    report_to="wandb",
    warmup_steps=30,
)

from transformers import DataCollatorForSeq2Seq
data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True, return_tensors="pt")

# Use only a small subset for quick testing
tokenized_dataset = tokenized_dataset.select(range(n_examples))

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    optimizers=(optimizer, scheduler)
)

trainer.train()
