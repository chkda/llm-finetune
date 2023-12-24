import os
import time
import wandb
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline, logging
from peft import LoraConfig
from trl import SFTTrainer

base_model = "NousResearch/Llama-2-7b-chat-hf"
dataset = "OdiaGenAI/odia_context_10K_llama2_set"
new_model = "odia-llama-7b"
repo = "Chhaya/odia-llama-7b"

dataset = load_dataset(dataset, split="train")

compute_dtype = getattr(torch, "float16")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map={"": 0}
)



model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM"
)
run_name = f"odia-llama-{int(time.time())}"

wandb.init(
    # set the wandb project where this run will be logged
    project="odia-llama-finetune",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 2e-4,
    "architecture": base_model,
    "dataset": dataset,
    "epochs": 1,
    },
    name=run_name
)


training_params = TrainingArguments(
    output_dir=repo,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_8bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard",
    push_to_hub=True
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_params,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_params,
    packing=False
)

trainer.train()

trainer.model.save_pretrained(repo)
trainer.tokenizer.save_pretrained(repo)

trainer.push_to_hub()


logging.set_verbosity(logging.CRITICAL)

prompt = "ଓଡ଼ିଶାରେ ଇକୋ-ଟୁରିଜିମକୁ ଆମେ କିପରି ପ୍ରୋତ୍ସାହିତ କରିପାରିବା?"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])
