import pandas as pd 
import model_utils
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
# llama = model_utils.OllamaModel(model_name = "llama3.2:3b-instruct-fp16")

# train, validation, test = pd.read_csv("./data/train.csv"), pd.read_csv("./data/validation.csv"), pd.read_csv("./data/test.csv") 
# system_prompt = """You are an experienced paraphrase detector, given the (text1) and (text2) below, determine if (text1) has been paraphrased to produce (text2). 
# *STRICTLY* format your output as only either "equivalent" OR "not equivalent" """
# correct = 0 
# for i in range(100):
#     text1, text2, label_text = test['text1'][i][:-2].strip(),test['text2'][i][:-2].strip(), test['label_text'][i].strip()
#     # print("the row is:", text1, text2, label_text)
    
#     response = llama.generate(prompt=f"text1: {text1}. text2: {text2}.", system = system_prompt)['response'].lower()
#     print(f"verdict: {response == label_text}. model: {response}, ground truth: {label_text}")
#     if response == label_text: correct += 1 
# print(f"the accuracy is {correct/100}")
    

