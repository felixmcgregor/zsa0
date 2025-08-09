#!/usr/bin/env python3
"""
Test to check if there's a mismatch between saved model and current config
"""

import torch
import os
from zsa0.nn import BottinaNet, ModelConfig

def test_model_loading():
    print("=== Testing Model Loading vs Current Config ===")
    
    # Current config from main.py
    current_config = ModelConfig(
        n_residual_blocks=1,
        conv_filter_size=32,  # This should create 12800 FC input
        n_policy_layers=4,
        n_value_layers=2,
        lr_schedule={0: 2e-3},
        l2_reg=4e-4,
    )
    
    print(f"Current config: {current_config}")
    
    # Create a new model with current config
    new_model = BottinaNet(current_config)
    print("✅ New model created successfully")
    
    # Check if there are any saved models in the training directory
    training_dir = "/home/felix/personal/zsa0/training"
    if os.path.exists(training_dir):
        for subdir in os.listdir(training_dir):
            model_path = os.path.join(training_dir, subdir, "model.pkl")
            if os.path.exists(model_path):
                print(f"\nFound saved model: {model_path}")
                try:
                    # Try to load the saved model
                    saved_model = torch.load(model_path, map_location='cpu')
                    print(f"✅ Loaded saved model successfully")
                    
                    # Check the config
                    if hasattr(saved_model, 'hparams'):
                        print(f"Saved model config: {saved_model.hparams}")
                        
                        # Check if conv_filter_size differs
                        if 'conv_filter_size' in saved_model.hparams:
                            saved_filters = saved_model.hparams['conv_filter_size']
                            print(f"Saved model conv_filter_size: {saved_filters}")
                            print(f"Current config conv_filter_size: {current_config.conv_filter_size}")
                            
                            if saved_filters != current_config.conv_filter_size:
                                print(f"🚨 MISMATCH FOUND!")
                                print(f"Saved model expects {saved_filters} filters")
                                print(f"Current config uses {current_config.conv_filter_size} filters")
                                
                                # Calculate what the saved model expects
                                saved_fc_size = saved_filters * 20 * 20
                                current_fc_size = current_config.conv_filter_size * 20 * 20
                                print(f"Saved model FC size: {saved_fc_size}")
                                print(f"Current FC size: {current_fc_size}")
                                
                                if saved_filters == 128:
                                    print("🎯 Found it! Saved model was trained with 128 filters = 2560x2560!")
                                
                        # Try to inspect the actual layer sizes
                        if hasattr(saved_model, 'fc_policy'):
                            first_linear = None
                            for module in saved_model.fc_policy:
                                if isinstance(module, torch.nn.Linear):
                                    first_linear = module
                                    break
                            
                            if first_linear:
                                input_size = first_linear.in_features
                                print(f"Saved model FC layer input size: {input_size}")
                                
                                # Calculate what filter size this corresponds to
                                implied_filters = input_size // (20 * 20)
                                print(f"This implies {implied_filters} conv filters")
                                
                                if input_size == 2560:
                                    print("🎯 Found it! FC layer expects 2560 inputs = 128 filters * 20!")
                                elif input_size == 2560 * 2560:
                                    print("🎯 Found it! FC layer is 2560x2560 matrix!")
                    
                except Exception as e:
                    print(f"❌ Error loading saved model: {e}")
                
                break
    else:
        print("No training directory found")

if __name__ == "__main__":
    test_model_loading()