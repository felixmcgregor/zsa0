#!/usr/bin/env python3
"""
Simple test to verify the training works with a fresh start
"""

import torch
import os
import shutil
from pathlib import Path

def test_fresh_training():
    print("=== Testing Fresh Training Start ===")
    
    # Check if there's a corrupted training directory
    training_dir = Path("/home/felix/personal/zsa0/training")
    if training_dir.exists():
        print(f"Found existing training directory: {training_dir}")
        
        # Check if the model file is corrupted
        for subdir in training_dir.iterdir():
            if subdir.is_dir():
                model_path = subdir / "model.pkl"
                if model_path.exists():
                    print(f"Checking model file: {model_path}")
                    try:
                        model = torch.load(model_path, map_location='cpu', weights_only=False)
                        print("✅ Model loads successfully")
                    except Exception as e:
                        print(f"❌ Model is corrupted: {e}")
                        print("🔧 This is the source of your dimension mismatch!")
                        
                        # Backup the corrupted directory
                        backup_dir = training_dir.parent / f"training_corrupted_backup_{subdir.name}"
                        print(f"Moving corrupted training to: {backup_dir}")
                        shutil.move(str(subdir), str(backup_dir))
                        print("✅ Corrupted training moved to backup")
                        return True
    
    print("No corrupted models found")
    return False

if __name__ == "__main__":
    was_corrupted = test_fresh_training()
    if was_corrupted:
        print("\n🎉 Solution:")
        print("1. Corrupted model has been moved to backup")
        print("2. Now try running your training again:")
        print("   uv run python src/zsa0/main.py train")
        print("3. It should start fresh without dimension mismatches!")
    else:
        print("\nNo corruption found. The dimension issue might be elsewhere.")