from unsloth import FastModel

model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/Qwen3-1.7B",
    max_seq_length=2048,
    load_in_4bit=True,
    load_in_8bit=False,
    full_finetuning=False,  # Set to True if performing full fine-tuning
)

from unsloth import LoRAConfig

lora_config = LoRAConfig(
    r=32,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

from unsloth import prepare_dataset

dataset = prepare_dataset(
    path="./converted/train.json", 
    tokenizer=tokenizer,
    chat_template="qwen3"
)

from unsloth import TrainerConfig

trainer_config = TrainerConfig(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    lora_config=lora_config,
    output_dir="qwen3_1.7b_lora_finetuned",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=500,
    evaluation_strategy="steps",
    save_total_limit=2,
    fp16=True,
    push_to_hub=False
)

trainer = trainer_config.get_trainer()
trainer.train()


model.save_pretrained("qwen3_1.7b_lora_finetuned")
tokenizer.save_pretrained("qwen3_1.7b_lora_finetuned")
