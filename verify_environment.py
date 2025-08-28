#!/usr/bin/env python3
"""
Environment verification script for GTA1 project.
This script checks if all required dependencies are properly installed.
"""

import sys
import importlib
import subprocess

def check_package(package_name, import_name=None):
    """Check if a package is installed and importable."""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name}")
        return True
    except ImportError:
        print(f"❌ {package_name} - Not installed or not importable")
        return False

def check_torch_acceleration():
    """Check available acceleration (CUDA, MPS, or CPU)."""
    try:
        import torch
        import platform
        
        system = platform.system()
        machine = platform.machine()
        
        if system == "Darwin":  # macOS
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                print(f"✅ PyTorch MPS support - Apple Silicon GPU acceleration available")
                print(f"   Device: {machine}")
                return True
            else:
                print(f"⚠️ PyTorch MPS support - Not available (CPU only)")
                print(f"   Device: {machine}")
                if machine == "arm64":
                    print("   Note: MPS should be available on Apple Silicon. Check PyTorch version.")
                return False
        else:  # Linux/Windows
            if torch.cuda.is_available():
                print(f"✅ PyTorch CUDA support - {torch.cuda.device_count()} GPU(s) available")
                print(f"   Current device: {torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'N/A'}")
                return True
            else:
                print("⚠️ PyTorch CUDA support - No CUDA GPUs available")
                return False
    except Exception as e:
        print(f"❌ PyTorch acceleration check failed: {e}")
        return False

def main():
    print("🔍 GTA1 Environment Verification")
    print("=" * 40)
    
    # Core packages
    packages = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("transformers", "transformers"),
        ("accelerate", "accelerate"),
        ("datasets", "datasets"),
        ("peft", "peft"),
        ("trl", "trl"),
        ("PIL (Pillow)", "PIL"),
        ("opencv-python", "cv2"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("tqdm", "tqdm"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("sklearn", "sklearn"),
        ("yaml", "yaml"),
        ("ultralytics", "ultralytics"),
        ("supervision", "supervision"),
        ("easyocr", "easyocr"),
        ("einops", "einops"),
        ("timm", "timm"),
        ("fire", "fire"),
        ("joblib", "joblib"),
    ]
    
    all_good = True
    
    print("\n📦 Checking core packages:")
    for package_name, import_name in packages:
        if not check_package(package_name, import_name):
            all_good = False
    
    print("\n🚀 Checking GPU support:")
    check_torch_acceleration()
    
    print("\n" + "=" * 40)
    if all_good:
        print("🎉 All packages are installed correctly!")
    else:
        print("⚠️ Some packages are missing. Run 'pip install -r requirements.txt' to install them.")
    
    print(f"\nPython version: {sys.version}")
    print(f"Virtual environment: {sys.prefix}")

if __name__ == "__main__":
    main()
