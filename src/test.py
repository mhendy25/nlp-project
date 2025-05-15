import pandas as pd 
import model_utils
import evaluate

def format_example(ex):
    label_text = "Yes" if ex["label"] == 1 else "No"
    text = (
        "Determine whether the following two sentences are paraphrases of each other. Respond with 'Yes' if they are paraphrases, and 'No' otherwise.\n"
        f"Sentence1: {ex['text1']}\n"
        f"Sentence2: {ex['text2']}\n"
        f"Paraphrase: {label_text}."
    )
    return {"text": text}
llama = model_utils.OllamaModel(model_name = "llama3.2:1b-instruct-fp16")

train, validation, test = pd.read_csv("./data/train.csv"), pd.read_csv("./data/validation.csv"), pd.read_csv("./data/test.csv") 
validation_data = validation.apply(lambda x: format_example(x), axis = 1)
predictions, references = [], []
print("the length of the set is", len(validation_data))
for i in range(len(validation_data)):
    if i%10==0:print(f"processing {i}/{len(validation_data)}")
    prompt = validation_data[i]["text"].rsplit("Paraphrase:", 1)[0] + "Paraphrase:"
    # print("the prompt is", prompt)
    response = llama.generate(prompt=prompt)['response'].lower().split("paraphrase:")[-1].strip().split()[0]
    # print("the resposne is", response)
    predictions.append(1 if response.startswith("yes") else 0)
    references.append(validation["label"][i])
metric = evaluate.load("glue", "mrpc")
result = metric.compute(predictions=predictions, references=references)
print("the results for the normal llama are:\n", result)