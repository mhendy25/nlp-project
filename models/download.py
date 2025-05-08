from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "meta-llama/Meta-Llama-3-3B"

tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,  # Or bfloat16 if on A100/H100
)

# Prepare a small test set
test_prompts = [
    "What is the capital of France?",
    "Explain the theory of relativity in simple terms.",
]

# Run inference
for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=100)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
model.save_pretrained("./llama3-local")
tokenizer.save_pretrained("./llama3-local")
