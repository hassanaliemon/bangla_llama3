# Llama-3 Finetune on Bangla Alpaca-LoRA Dataset 🦙

Welcome to the LLaMA3 Fine-tuning repository! This project focuses on fine-tuning the LLaMA3 language model using the Bangla Alpaca dataset. By leveraging state-of-the-art techniques in natural language processing (NLP), we aim to enhance the model's proficiency in understanding and generating text in the Bengali language.

## Overview
LLaMA3 (Large Language Model) is a powerful language model known for its ability to generate human-like text based on the input it receives. Fine-tuning LLaMA3 on specific datasets allows the model to better understand the nuances and unique characteristics of particular languages or tasks. In this project, we fine-tune LLaMA3 on the Bangla Alpaca data

## Local Setup

### Install dependencies

```bash
pip install -r requirements.txt
```


### Training (`finetune.py`)

To finetune run the below command

```bash
python finetune.py \
    --base_model 'NousResearch/Meta-Llama-3-8B' \
    --data_path 'BanglaLLM/bangla-alpaca' \
    --output_dir './lora-alpaca'
```

To tweak hyperparameters:

```bash
python finetune.py \
    --base_model 'NousResearch/Meta-Llama-3-8B' \
    --data_path 'BanglaLLM/bangla-alpaca' \
    --output_dir './lora-alpaca
    --batch_size 128 \
    --micro_batch_size 4 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --val_set_size 2000 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs \
    --group_by_length
```
Note: you can provide one or more from [q_proj,k_proj,v_proj,o_proj] to `lora_target_modules` parameter. It will increase the trainable parameters but model's outcome will get better as well

### Resume Training
To resume training from the checkpoint, provide the checkpoint path
```bash
python finetune.py \
    --base_model 'NousResearch/Meta-Llama-3-8B' \
    --data_path 'BanglaLLM/bangla-alpaca' \
    --output_dir './lora-alpaca',\
    --resume_from_checkpoint './lora-alpaca'
```


### Inference

To inference, specify the `base_model` and `lora_weights` path of `generate.py` file at line `62, 63`, then run the following command 
```bash
python generate.py
```
