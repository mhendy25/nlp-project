import pandas as pd 
import model_utils

llama = model_utils.OllamaModel(model_name = "llama3.2:1b")

train, validation, test = pd.read_csv("./data/train.csv"), pd.read_csv("./data/validation.csv"), pd.read_csv("./data/test.csv") 
correct = 0 
for i in range(100):
    text1, text2, label_text = test['text1'][i][:-2],test['text2'][i][:-2].strip(), test['label_text'][i].strip()
    # print("the row is:", text1, text2, label_text)
    system_prompt = """You are an experienced paraphrase detector, given the (text1) and (text2) below, determine if (text1) has been paraphrased to produce (text2). 
*STRICTLY* format your output as only: "equivalent" OR "not equivalent" otherwise.
text1: {text1}.
text2: {text2}."""
    response = llama.generate(prompt=system_prompt)
    print(f"model: {response['response']}, ground truth: {label_text}")
    if response['response'] == label_text: correct += 1 
print(f"the accuracy is {correct/100}")
    

