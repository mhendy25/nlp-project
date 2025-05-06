from datasets import load_dataset
import os
dataset = load_dataset("SetFit/mrpc")

os.makedirs("./data", exist_ok=True)

# print(dataset)
for split, data in dataset.items():
    df = data.to_pandas()
    df = df.rename(columns={"sentence1": "text1", "sentence2": "text2", "label": "label_text"})
    df.to_csv(f"./data/{split}.csv", index=False)