#!/usr/bin/env python3
"""
Debug Script for Translation Failures
=====================================

Helps diagnose why translations are failing.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    load_dotenv(env_file)
    print(f"✓ Loaded .env file")

print("\n" + "=" * 80)
print("INDICTRANS2 DEBUG DIAGNOSTICS")
print("=" * 80)

# 1. Check token
print("\n1. CHECKING HUGGINGFACE TOKEN")
print("-" * 80)
token = os.getenv('HF_TOKEN')
if token:
    print(f"✓ Token found: {token[:20]}...")
else:
    print("✗ No token in .env")
    print("  Create .env with: echo 'HF_TOKEN=hf_...' > .env")

# 2. Check memory
print("\n2. CHECKING SYSTEM MEMORY")
print("-" * 80)
try:
    import psutil

    memory = psutil.virtual_memory()
    print(f"Total RAM: {memory.total / (1024 ** 3):.2f} GB")
    print(f"Available: {memory.available / (1024 ** 3):.2f} GB")
    print(f"Used: {memory.used / (1024 ** 3):.2f} GB")
    print(f"Percent: {memory.percent}%")

    if memory.available / (1024 ** 3) < 2:
        print("⚠️  WARNING: Less than 2GB available - model may fail!")
except ImportError:
    print("psutil not installed, skipping memory check")
    print("Install: pip install psutil")

# 3. Check dependencies
print("\n3. CHECKING DEPENDENCIES")
print("-" * 80)

deps = {
    'torch': 'PyTorch',
    'transformers': 'Transformers',
    'sentencepiece': 'SentencePiece',
    'huggingface_hub': 'HuggingFace Hub',
}

all_ok = True
for module, name in deps.items():
    try:
        mod = __import__(module)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✓ {name}: {version}")
    except ImportError:
        print(f"✗ {name}: NOT INSTALLED")
        all_ok = False

if not all_ok:
    print("\nInstall missing packages:")
    print("  pip install torch transformers sentencepiece --break-system-packages")

# 4. Test model loading
print("\n4. TESTING MODEL LOADING")
print("-" * 80)

try:
    print("Importing transformers...")
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    print("✓ Imports successful")

    print("\nLoading tokenizer...")
    model_name = "ai4bharat/indictrans2-indic-en-1B"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=False
    )
    print("✓ Tokenizer loaded")

    print("\nLoading model (this may take a minute)...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    print("✓ Model loaded successfully")

    # 5. Test translation
    print("\n5. TESTING TRANSLATION")
    print("-" * 80)

    import torch

    test_hindi = "नमस्ते"  # Simple Hindi word: "Namaste"
    print(f"Test input (Hindi): {test_hindi}")

    try:
        input_text = f"hin_Deva {test_hindi}"
        print(f"Formatted input: {input_text}")

        inputs = tokenizer(
            input_text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        print(f"✓ Tokenized: input_ids shape = {inputs['input_ids'].shape}")

        print("\nGenerating translation...")
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                num_beams=4,
                max_length=512,
                num_return_sequences=1,
                temperature=0.7,
            )

        translation = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"✓ Translation successful!")
        print(f"  Output: {translation}")

        if translation and translation.strip() and translation != "[TRANSLATION FAILED]":
            print("\n✅ TRANSLATION WORKS!")
        else:
            print("\n⚠️  Translation generated but empty/invalid")

    except Exception as e:
        print(f"✗ Translation failed: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()

except Exception as e:
    print(f"✗ Model loading failed: {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()

    print("\n" + "=" * 80)
    print("SOLUTIONS")
    print("=" * 80)
    print("\n1. Install all dependencies:")
    print("   pip install torch transformers sentencepiece huggingface-hub --break-system-packages")
    print("\n2. Authenticate with HuggingFace:")
    print("   huggingface-cli login")
    print("   OR add token to .env:")
    print("   echo 'HF_TOKEN=hf_...' > .env")
    print("\n3. Check available memory:")
    print("   free -h")
    print("\n4. If memory low, try on a machine with more RAM")

print("\n" + "=" * 80)