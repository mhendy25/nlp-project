import pandas as pd

# Read the .txt file
d1 = pd.read_csv(
    '/Users/stephenlee/.cache/kagglehub/datasets/doctri/microsoft-research-paraphrase-corpus/versions/1/msr_paraphrase_train.txt',
    sep='\t', header=0, on_bad_lines='skip'
)

# Rename the columns
d1 = d1.rename(columns={'#1 String': 'text1', '#2 String': 'text2', 'Quality': 'label_text'})

# Save as CSV
d1.to_csv('train.csv', index=False)


# Read the .txt file
d1 = pd.read_csv(
    '/Users/stephenlee/.cache/kagglehub/datasets/doctri/microsoft-research-paraphrase-corpus/versions/1/msr_paraphrase_test.txt',
    sep='\t', header=0, on_bad_lines='skip'
)

# Rename the columns
d1 = d1.rename(columns={'#1 String': 'text1', '#2 String': 'text2', 'Quality': 'label_text'})

# Save as CSV
d1.to_csv('test.csv', index=False)

# Read the .txt file
d1 = pd.read_csv(
    '/Users/stephenlee/.cache/kagglehub/datasets/doctri/microsoft-research-paraphrase-corpus/versions/1/msr_paraphrase_train.txt',
    sep='\t', header=0, on_bad_lines='skip'
)

# Rename the columns
d1 = d1.rename(columns={'#1 String': 'text1', '#2 String': 'text2', 'Quality': 'label_text'})

# Save as CSV
d1.to_csv('test.csv', index=False)