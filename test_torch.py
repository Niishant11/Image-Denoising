#!/usr/bin/env python
"""Test PyTorch installation"""
import sys
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ CUDA device: {torch.cuda.get_device_name(0)}")
    print("\n✓ PyTorch import successful!")
    sys.exit(0)
except Exception as e:
    print(f"✗ Error importing PyTorch: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
