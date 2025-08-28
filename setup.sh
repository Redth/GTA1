#!/bin/bash

# Setup script for GTA1 project
# This script creates a Python virtual environment and installs all dependencies
# Supports both Linux/WSL2 (CUDA) and macOS (CPU/MPS)

set -e  # Exit on any error

echo "🚀 Setting up GTA1 project environment..."

# Detect platform
OS="$(uname -s)"
ARCH="$(uname -m)"

echo "🖥️  Detected platform: $OS ($ARCH)"

# Create virtual environment
echo "📦 Creating Python virtual environment..."
python3 -m venv .venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Platform-specific PyTorch installation
if [[ "$OS" == "Darwin" ]]; then
    echo "🍎 Installing PyTorch for macOS..."
    if [[ "$ARCH" == "arm64" ]]; then
        echo "   Apple Silicon detected - installing with MPS support"
        pip install torch torchvision torchaudio
    else
        echo "   Intel Mac detected - installing CPU version"
        pip install torch torchvision torchaudio
    fi
elif [[ "$OS" == "Linux" ]]; then
    echo "🐧 Installing PyTorch for Linux with CUDA support..."
    pip install torch torchvision==0.19.0+cu121 --index-url https://download.pytorch.org/whl/cu121
else
    echo "⚠️  Unknown platform, installing default PyTorch..."
    pip install torch torchvision torchaudio
fi

# Install remaining requirements
echo "📚 Installing remaining dependencies..."
if [[ "$OS" == "Darwin" ]]; then
    echo "   Using macOS-compatible package versions..."
    # Skip problematic packages on macOS and install the rest
    pip install transformers>=4.44 accelerate>=0.33 datasets peft>=0.11 "trl>=0.14,<0.15"
    pip install pillow opencv-python numpy pandas tqdm matplotlib seaborn scikit-learn packaging pyyaml
    pip install sentencepiece protobuf ultralytics==8.3.70 supervision==0.18.0
    pip install easyocr einops==0.8.0 timm fire joblib imagesize
    
    echo "   ⚠️  Skipping macOS-incompatible packages:"
    echo "      - bitsandbytes (not available on macOS)"
    echo "      - liger-kernel (may have compatibility issues)"
    echo "      - opencv-python-headless (can conflict with opencv-python)"
else
    pip install -r requirements.txt
fi

echo "✅ Setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "  source .venv/bin/activate"
echo ""
if [[ "$OS" == "Darwin" ]]; then
    echo "🍎 macOS Notes:"
    if [[ "$ARCH" == "arm64" ]]; then
        echo "  - GPU acceleration available via Metal Performance Shaders (MPS)"
        echo "  - Use 'mps' device instead of 'cuda' in training scripts"
    else
        echo "  - CPU-only training (no GPU acceleration available)"
    fi
    echo "  - Training will be slower than on CUDA GPUs"
    echo "  - Consider reducing batch size and model size for testing"
else
    echo "To run the training, use:"
    echo "  export CUDA_VISIBLE_DEVICES=0"
    echo "  ./run_training.sh"
fi
