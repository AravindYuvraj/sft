# sft

# Tiny Supervised Fine-Tuning (SFT) – Aligning a Base LLM

This project demonstrates how to fine-tune a small language model using only 20–30 prompt-response examples to align it for:

- Politeness
- Factual accuracy
- Answer length control
- Safe refusals for harmful inputs

## Model

- **Base model:** `tiiuae/falcon-rw-1b`
- **Fine-tuning method:** LoRA (via PEFT)
- **Trained with:** Transformers, Datasets
- **Environment:** Google Colab

## Contents

- `train.py`: Fine-tunes the base model using LoRA
- `prompts.jsonl`: Training dataset (20 examples)
- `before_after.md`: Side-by-side comparison of model outputs before and after fine-tuning
- `README.md`: You're here!

## Result Summary

The fine-tuned model:
- Answers more clearly and politely
- Responds with appropriate length
- Refuses unsafe prompts reliably

This proves small-scale fine-tuning can align a base model effectively.
