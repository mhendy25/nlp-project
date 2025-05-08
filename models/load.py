from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained("./Meta-Llama-3-3B", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("./Meta-Llama-3-3B")


# Prepare a small test set
test_prompts = [
    "What is the capital of France?",
    "Explain the theory of relativity in simple terms.",
]

# Run inference
for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=100)
    print("after loading:", tokenizer.decode(outputs[0], skip_special_tokens=True))