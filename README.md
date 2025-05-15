# Evaluating Locally Deployable LLMs on the Microsoft Research Paraphrase Corpus

A project for fine-tuning and evaluating *light-weight* language models on the Microsoft Research Paraphrase Corpus (MRPC) for paraphrase detection.

## Project Overview

This project implements:
- Loading and preprocessing of the MRPC dataset
- Fine-tuning of language models (Llama 3.2 and Qwen)
- Model evaluation on paraphrase detection
- Integration with Ollama for local model inference

## Project Structure

```
├── data/                  # Dataset files
│   ├── train.csv          # Training data
│   ├── validation.csv     # Validation data
│   └── test.csv           # Test data
├── fine-tuning/           # Notebooks for model fine-tuning
│   ├── llama_fine_tuning_with_unsloth_apis.ipynb
│   └── qwen_fine_tuning_with_unsloth_apis.ipynb
├── src/                   # Source code
│   ├── load_dataset.py    # Script to load and prepare dataset
│   ├── model_utils.py     # Utilities for model interaction via Ollama
│   └── test.py            # Evaluation script
├── requirements.txt       # Project dependencies
└── .gitignore             # Git ignore file
```

## Setup and Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd <repository-name>
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Install Ollama following the instructions at [https://ollama.com/](https://ollama.com/)

## Data Preparation

The project uses the Microsoft Research Paraphrase Corpus (MRPC) dataset from Hugging Face:

```
python src/load_dataset.py
```

This will download the dataset and convert it to CSV format in the `data/` directory.

## Running the Model

1. Make sure Ollama is running on your machine
2. Pull the required model (default is Llama 3.2 1B):
   ```
   ollama pull llama3.2:1b-instruct-fp16
   ```
3. Run the evaluation script:
   ```
   python src/test.py
   ```

## Fine-tuning

The project includes Jupyter notebooks for fine-tuning language models using the Unsloth library:
- `fine-tuning/llama_fine_tuning_with_unsloth_apis.ipynb` - For fine-tuning Llama models
- `fine-tuning/qwen_fine_tuning_with_unsloth_apis-2.ipynb` - For fine-tuning Qwen models

To use these notebooks:
1. Open them in Jupyter or a compatible environment (Google Colab recommended)
2. Follow the steps in the notebooks to fine-tune models
3. Export the fine-tuned models for use with Ollama

## Customization

- To use a different model, modify the model name in `src/test.py`
- To adjust generation parameters, modify the `generate` method call in `src/model_utils.py`
- To change the evaluation metrics, update the evaluation code in `src/test.py`

## Requirements

- Python 3.8+
- Ollama (for local inference)
- Dependencies listed in `requirements.txt`
- GPU recommended for fine-tuning (not required for inference with Ollama)
