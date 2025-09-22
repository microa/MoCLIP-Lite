#!/usr/bin/env python3
"""
Example usage script for MoCLIP-Lite.
This script demonstrates how to use the different components of the framework.
"""

import sys
import os
import torch

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model import MV_TSN_Model
from data.dataloader import UCF101Dataset
from data.transforms_video import GroupCenterSample
from utils.generate_text_features import main as generate_text_features
from utils.precompute_clip_features import main as precompute_clip_features

def example_model_usage():
    """Example of how to use the MV-TSN model."""
    print("=== MV-TSN Model Usage Example ===")
    
    # Initialize model
    model = MV_TSN_Model(num_classes=101, base_model='efficientnet_b0')
    model.eval()
    
    # Create dummy input (batch_size=1, channels=2, height=224, width=224)
    dummy_input = torch.randn(1, 2, 224, 224)
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of classes: {output.shape[1]}")
    print("✓ Model usage example completed\n")

def example_data_loading():
    """Example of how to use the data loaders."""
    print("=== Data Loading Example ===")
    
    # Example configuration
    data_root = "/path/to/ucf101/videos"
    list_file = "/path/to/ucf101/test_list.txt"
    
    print(f"Data root: {data_root}")
    print(f"List file: {list_file}")
    print("Note: Update these paths to your actual data locations")
    print("✓ Data loading example completed\n")

def example_feature_generation():
    """Example of how to generate text and CLIP features."""
    print("=== Feature Generation Example ===")
    print("To generate text features:")
    print("  python utils/generate_text_features.py")
    print()
    print("To precompute CLIP features:")
    print("  python utils/precompute_clip_features.py")
    print("✓ Feature generation example completed\n")

def main():
    """Main example function."""
    print("MoCLIP-Lite Usage Examples")
    print("=" * 50)
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print()
    
    # Run examples
    example_model_usage()
    example_data_loading()
    example_feature_generation()
    
    print("=" * 50)
    print("All examples completed!")
    print("\nFor more detailed usage, see the README.md file.")

if __name__ == "__main__":
    main()
