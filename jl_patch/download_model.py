#!/usr/bin/env python3
"""
Script to download Qwen2.5-0.5B-Instruct model for GSM8K training
Usage: python3 jl_patch/download_model.py
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def main():
    model_name = 'Qwen/Qwen2.5-0.5B-Instruct'
    save_dir = '/mnt/hdfs/tiktok_aiic/user/junlongli/models/Qwen2.5-0.5B-Instruct'

    print(f'Downloading model: {model_name}')
    print(f'Target directory: {save_dir}')
    print('-' * 60)

    # Download tokenizer
    print('Downloading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Download model
    print('Downloading model weights...')
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Save to target directory
    print(f'\nSaving to {save_dir}...')
    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(save_dir)

    print('\n✓ Model download completed!')
    print(f'Model saved at: {save_dir}')

if __name__ == '__main__':
    main()
