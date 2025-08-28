#!/usr/bin/env python3
"""
Platform detection script for GTA1 project.
Helps determine the best setup approach for your system.
"""

import platform
import sys

def detect_platform():
    """Detect platform and provide setup recommendations."""
    system = platform.system()
    machine = platform.machine()
    python_version = platform.python_version()
    
    print("🔍 GTA1 Platform Detection")
    print("=" * 40)
    print(f"Operating System: {system}")
    print(f"Architecture: {machine}")
    print(f"Python Version: {python_version}")
    print()
    
    if system == "Darwin":  # macOS
        print("🍎 macOS Detected")
        if machine == "arm64":
            print("✅ Apple Silicon (M1/M2/M3) - MPS acceleration available")
            print("📋 Recommendations:")
            print("   • Use ./setup.sh for automatic setup")
            print("   • GPU acceleration via Metal Performance Shaders (MPS)")
            print("   • Training will be slower than CUDA but faster than CPU")
            print("   • Good for development and testing")
        else:
            print("🔧 Intel Mac - CPU only")
            print("📋 Recommendations:")
            print("   • Use ./setup.sh for automatic setup")
            print("   • No GPU acceleration available")
            print("   • Training will be very slow")
            print("   • Consider cloud training for serious work")
            
        print("\n⚠️  macOS Limitations:")
        print("   • No CUDA support (NVIDIA GPUs not supported)")
        print("   • Some packages may not be available (e.g., bitsandbytes)")
        print("   • Reduced training performance compared to Linux/CUDA")
        
    elif system == "Linux":
        print("🐧 Linux Detected")
        print("📋 Recommendations:")
        print("   • Use ./setup.sh for automatic setup")
        print("   • CUDA GPU acceleration available (if NVIDIA GPU present)")
        print("   • Original target platform - best performance")
        print("   • Full package compatibility")
        
    elif system == "Windows":
        print("🪟 Windows Detected")
        print("📋 Recommendations:")
        print("   • Consider using WSL2 (Windows Subsystem for Linux)")
        print("   • Native Windows training may have package compatibility issues")
        print("   • WSL2 provides Linux-like environment with CUDA support")
        
    else:
        print(f"❓ Unknown Platform: {system}")
        print("📋 Recommendations:")
        print("   • Try the standard setup.sh script")
        print("   • May need manual package installation")

    print(f"\n🐍 Python Version Check:")
    if sys.version_info >= (3, 8):
        print(f"   ✅ Python {python_version} is supported")
    else:
        print(f"   ❌ Python {python_version} is too old")
        print("   📋 Please upgrade to Python 3.8 or newer")

def main():
    detect_platform()
    
    print("\n" + "=" * 40)
    print("Next Steps:")
    print("1. Run './setup.sh' to create virtual environment")
    print("2. Run 'python verify_environment.py' to check installation")
    print("3. Run './run_training.sh' to start training (with appropriate data)")
    print("\nFor detailed instructions, see SETUP.md")

if __name__ == "__main__":
    main()
