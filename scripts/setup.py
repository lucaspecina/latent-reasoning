import subprocess
import sys

def check_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def main():
    # Uninstall existing torch
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torch"])
    
    # Install torch with appropriate version
    if check_cuda():
        print("Installing PyTorch with CUDA support...")
        subprocess.run([
            sys.executable, 
            "-m", 
            "pip", 
            "install", 
            "torch", 
            "--index-url", 
            "https://download.pytorch.org/whl/cu121"
        ])
    else:
        print("Installing CPU-only PyTorch...")
        subprocess.run([
            sys.executable, 
            "-m", 
            "pip", 
            "install", 
            "torch"
        ])
    
    # Install other requirements
    subprocess.run([
        sys.executable, 
        "-m", 
        "pip", 
        "install", 
        "-r", 
        "requirements.txt"
    ])

if __name__ == "__main__":
    main() 