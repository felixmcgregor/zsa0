#!/usr/bin/env python3
"""
Simple test script to verify the neural network dimensions work correctly
for the Zootopia game adaptation.
"""

import torch
import torch.nn as nn
from einops import rearrange

# Constants from the Rust module (hardcoded for testing)
BUF_N_CHANNELS = 3  # From Rust: 3 channels
GRID_HEIGHT = 20    # From Rust: DEFAULT_HEIGHT = 20
GRID_WIDTH = 20     # From Rust: DEFAULT_WIDTH = 20
N_COLS = 4          # From Rust: N_MOVES = 4 (up, down, left, right)

class SimpleResidualBlock(nn.Module):
    def __init__(self, n_channels: int, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, padding=padding),
            nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return x + self.block(x)

class TestNet(nn.Module):
    """Simplified version of BottinaNet for testing dimensions."""
    
    def __init__(self, conv_filter_size=32, n_residual_blocks=1):  # Use exact config from main.py
        super().__init__()
        
        # Convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(BUF_N_CHANNELS, conv_filter_size, kernel_size=3, padding=1),
            *[SimpleResidualBlock(conv_filter_size) for _ in range(n_residual_blocks)],
        )
        
        # Calculate the size after conv layers
        fc_size = self._calculate_conv_output_size()
        print(f"Calculated FC size: {fc_size}")
        
        # Policy head (simplified) - using n_policy_layers=4 from main.py
        self.fc_policy = nn.Sequential(
            nn.Linear(fc_size, fc_size),
            nn.BatchNorm1d(fc_size),
            nn.ReLU(),
            nn.Linear(fc_size, fc_size),
            nn.BatchNorm1d(fc_size),
            nn.ReLU(),
            nn.Linear(fc_size, fc_size),
            nn.BatchNorm1d(fc_size),
            nn.ReLU(),
            nn.Linear(fc_size, N_COLS),
            nn.LogSoftmax(dim=1),
        )
        
        # Value head (simplified) - using n_value_layers=2 from main.py
        self.fc_value = nn.Sequential(
            nn.Linear(fc_size, fc_size),
            nn.BatchNorm1d(fc_size),
            nn.ReLU(),
            nn.Linear(fc_size, 2),  # q_penalty, q_no_penalty
            nn.Tanh(),
        )
    
    def _calculate_conv_output_size(self):
        """Calculate the output size of the convolutional block."""
        dummy_input = torch.zeros(1, BUF_N_CHANNELS, GRID_HEIGHT, GRID_WIDTH)
        print(f"Input shape: {dummy_input.shape}")
        
        with torch.no_grad():
            dummy_output = self.conv(dummy_input)
        
        print(f"Conv output shape: {dummy_output.shape}")
        flattened_size = dummy_output.numel() // dummy_output.size(0)
        print(f"Flattened size per batch item: {flattened_size}")
        return flattened_size
    
    def forward(self, x):
        print(f"Forward input shape: {x.shape}")
        
        # Conv layers
        x = self.conv(x)
        print(f"After conv shape: {x.shape}")
        
        # Flatten
        x = rearrange(x, "b c h w -> b (c h w)")
        print(f"After flatten shape: {x.shape}")
        
        # Policy and value heads
        policy_logprobs = self.fc_policy(x)
        q_values = self.fc_value(x)
        
        print(f"Policy output shape: {policy_logprobs.shape}")
        print(f"Q values output shape: {q_values.shape}")
        
        return policy_logprobs, q_values

def test_network():
    """Test the network with sample input."""
    print("="*50)
    print("Testing Zootopia Neural Network Dimensions")
    print("Using EXACT config from main.py:")
    print("- conv_filter_size: 32")
    print("- n_residual_blocks: 1") 
    print("- n_policy_layers: 4")
    print("- n_value_layers: 2")
    print("="*50)
    
    # Create network with exact config from main.py
    net = TestNet(conv_filter_size=32, n_residual_blocks=1)
    
    # Test with batch of 2 samples
    batch_size = 2
    test_input = torch.randn(batch_size, BUF_N_CHANNELS, GRID_HEIGHT, GRID_WIDTH)
    
    print(f"\nTesting with batch size: {batch_size}")
    print(f"Input tensor shape: {test_input.shape}")
    print(f"Expected: ({batch_size}, {BUF_N_CHANNELS}, {GRID_HEIGHT}, {GRID_WIDTH})")
    
    try:
        # Forward pass
        policy, q_values = net(test_input)
        
        print(f"\n✅ SUCCESS!")
        print(f"Policy shape: {policy.shape} (expected: ({batch_size}, {N_COLS}))")
        print(f"Q values shape: {q_values.shape} (expected: ({batch_size}, 2))")
        
        # Test if shapes are correct
        assert policy.shape == (batch_size, N_COLS), f"Policy shape mismatch: {policy.shape} != ({batch_size}, {N_COLS})"
        assert q_values.shape == (batch_size, 2), f"Q values shape mismatch: {q_values.shape} != ({batch_size}, 2)"
        
        print("\n✅ All dimension checks passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_network()
    if success:
        print("\n🎉 Neural network dimensions are working correctly!")
    else:
        print("\n💥 Neural network has dimension issues!")