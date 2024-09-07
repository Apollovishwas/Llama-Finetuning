# Fintuning Llama 2 with Open Platypus Dataset to imporove Logical Reasoning
## Dataset Preparation for Fine-Tuning
This script prepares the Open-Platypus dataset for fine-tuning a language model, specifically tailored for the Llama 2 model.
## Dataset: Open-Platypus 
####  HugginFace : https://huggingface.co/datasets/garage-bAInd/Open-Platypus
The Open-Platypus dataset is a collection of instruction-output pairs used for training language models. It is loaded from the Hugging Face datasets library
### Key Processing Steps:
1. Token Count Analysis: Analyzes the distribution of token counts in instructions and outputs,
2.Data Filtering: Removes entries exceeding 2048 tokens to fit Llama 2's context window,
3.Near-Deduplication: Uses sentence embeddings to remove near-duplicate entries, ensuring dataset diversity,
4.Top-k Sampling: Selects the 1000 samples with the highest token counts,
5.Chat Template Formatting: Applies a specific format to the instruction field for fine-tuning,

Output
1. The processed dataset is exported as "Open-Platypus-Filtered-FineTuning" to the Hugging Face Hub
2. This preparation ensures a high-quality, diverse dataset optimized for fine-tuning language models like Llama 2.
   ####   Output Dataset : https://huggingface.co/datasets/veechan/Open-Platypus-Filtered-FineTuning

## Llama-2-7b Fine-Tuning
This script fine-tunes the Llama-2-7b model on the mini-platypus dataset using quantization and LoRA techniques.
## Features
Uses 4-bit quantization to reduce memory footprint
Implements LoRA for efficient fine-tuning
Fine-tunes on the mini-platypus dataset
Saves and evaluates the fine-tuned model
Optionally pushes the model to Hugging Face Hub
### Requirements
Python 3.7+,
PyTorch,
Transformers,
Datasets,
Accelerate,
PEFT,
TRL,
Bitsandbytes,
Wandb,
### Usage
Install dependencies: pip install -q -U transformers datasets accelerate peft trl bitsandbytes wandb
Run the script: python fine_tuning.py
### Note
To run the output model, you'll need at least 24GB of VRAM.
### Output
####  Finetuned Model : https://huggingface.co/veechan/llama-2-7b-platypus-finetuned

The script produces a fine-tuned version of Llama-2-7b, saved as "llama-2-7b-miniplatypus".
