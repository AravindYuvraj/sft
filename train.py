from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer
import torch

# Lightweight Model
model_id = "tiiuae/falcon-rw-1b"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Load dataset
dataset = load_dataset("json", data_files="dataset.jsonl", split="train")

# Tokenize function (fixed version)
def tokenize(example):
    # Flatten the messages into a prompt/response format
    user_msg = ""
    assistant_msg = ""

    for turn in example["messages"]:
        if turn["role"] == "user":
            user_msg += turn["content"] + "\n"
        elif turn["role"] == "assistant":
            assistant_msg += turn["content"] + "\n"

    full_prompt = f"User: {user_msg.strip()}\nAssistant: {assistant_msg.strip()}"

    # Tokenize it
    tokenized = tokenizer(
        full_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    return {
        "input_ids": tokenized["input_ids"][0],
        "attention_mask": tokenized["attention_mask"][0]
    }


# Tokenize dataset
tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

# LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# Wrap model with LoRA
model = get_peft_model(model, lora_config)

# Training configuration
args = TrainingArguments(
    output_dir="./sft_model",
    per_device_train_batch_size=1,
    num_train_epochs=3,
    learning_rate=5e-5,
    fp16=True,
    logging_steps=10,
    save_total_limit=1,
    save_steps=100
)

# Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=args
)

import os
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"


# Start training!
trainer.train()