from datasets import load_dataset

dataset = load_dataset("SetFit/mrpc")
# print(dataset)
for split, data in dataset.items():
    data.to_csv(f"./data/{split}.csv", index = None)
    