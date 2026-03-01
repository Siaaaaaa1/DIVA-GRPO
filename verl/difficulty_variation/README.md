# Difficulty Variation Pipeline

This pipeline enhances datasets by using an LLM (Qwen-VL) to generate **Variants** and **Think Steps** for original problems. 

## Features

* **Streamlined Processing**: Uses a **Producer-Consumer** architecture. Multiple workers generate data in parallel, while a single listener process continuously streams the results into **one single output file**.
* **Memory Efficient**: Data is flushed to disk every 100 records (Row Groups). This prevents memory overflow (OOM) even with large datasets.
* **Quality Control**: Automatically verifies that the generated reasoning steps (`think_steps`) lead to the correct answer (`\boxed{answer}`) before saving.

## Requirements

Ensure dependencies are installed (`pyarrow`, `pandas`, etc.) and configure your Azure OpenAI environment variables:

```bash
export AZURE_OPENAI_KEY="your_key"
export AZURE_OPENAI_ENDPOINT="your_endpoint"
```

Usage
Run augment_dataset.py to start processing:


```bash
python verl/difficulty_variation/augment_dataset.py \
  --input "path/to/input.parquet" \
  --output "path/to/output.parquet" \
  --workers 8
```

Arguments
- --input: Path to the input Parquet dataset.

- --output: Full path to the final single output Parquet file (e.g., data/processed/final_dataset.parquet).

- --workers: Number of parallel worker processes.

Output Structure
The script generates a single Parquet file at the specified output path. The file contains the original columns plus:

- variants: (List[str]) 5 different wordings of the problem.

- think_steps: (List[str]) Step-by-step reasoning process.

- think_answer: (str) The final answer extracted from the reasoning.