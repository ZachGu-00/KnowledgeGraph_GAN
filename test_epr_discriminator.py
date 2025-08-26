#!/usr/bin/env python3
"""
Test script for EPR Discriminator
"""
import torch
from models.epr_discriminator import test_epr_discriminator

def main():
    print("=== Testing EPR Discriminator ===")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Run test
    test_epr_discriminator()

if __name__ == "__main__":
    main()