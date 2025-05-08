import pandas as pd
import json
import os

def convert_csv_to_unsloth_json(input_csv, output_json):
    df = pd.read_csv(input_csv)
    messages_list = []

    for i in range(len(df)):
        text1 = str(df['text1'][i]).strip()[:-2]
        text2 = str(df['text2'][i]).strip()[:-2]
        label_text = str(df['label_text'][i]).strip()

        # Normalize label
        if label_text == "1":
            label_text = "equivalent"
        elif label_text == "0":
            label_text = "not equivalent"

        system_prompt = f"""You are an experienced paraphrase detector, given the (text1) and (text2) below, determine if (text1) has been paraphrased to produce (text2). 
*STRICTLY* format your output as only: "equivalent" OR "not equivalent" otherwise.
text1: {text1}.
text2: {text2}."""

        messages = {
            "messages": [
                {"role": "user", "content": system_prompt},
                {"role": "assistant", "content": label_text}
            ]
        }
        messages_list.append(messages)

    # Save as JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(messages_list, f, ensure_ascii=False, indent=2)

# Convert all splits
os.makedirs("./converted", exist_ok=True)
convert_csv_to_unsloth_json("./data/train.csv", "./converted/train.json")
convert_csv_to_unsloth_json("./data/validation.csv", "./converted/validation.json")
convert_csv_to_unsloth_json("./data/test.csv", "./converted/test.json")
