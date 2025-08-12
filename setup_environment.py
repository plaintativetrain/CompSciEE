"""
Setup script for 3D Reconstruction Comparison Experiment
This script helps prepare the environment and download necessary repositories
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import requests
import zipfile
import tarfile

def run_command(command, description="", check=True):
    """Run a shell command with logging"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    try:
        if isinstance(command, str):
            result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        else:
            result = subprocess.run(command, check=check, capture_output=True, text=True)
        
        if result.stdout:
            print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        print(f"✅ SUCCESS: {description}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ FAILED: {description}")
        print(f"Error: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def check_system_requirements():
    """Check if system meets minimum requirements"""
    print("Checking system requirements...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python {python_version.major}.{python_version.minor}")
    
    # Check for CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        else:
            print("⚠️  CUDA not available - will use CPU (much slower)")
    except ImportError:
        print("⚠️  PyTorch not installed yet")
    
    # Check available disk space (needs at least 20GB)
    disk_usage = shutil.disk_usage(Path.cwd())
    free_gb = disk_usage.free / (1024**3)
    if free_gb < 20:
        print(f"❌ Insufficient disk space: {free_gb:.1f}GB available, need 20GB+")
        return False
    print(f"✅ Disk space: {free_gb:.1f}GB available")
    
    return True

def install_python_dependencies():
    """Install Python dependencies"""
    print("\nInstalling Python dependencies...")
    
    # Upgrade pip first
    if not run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      "Upgrading pip"):
        return False
    
    # Install requirements
    requirements_file = Path("requirements.txt")
    if requirements_file.exists():
        if not run_command([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)], 
                          "Installing requirements"):
            return False
    else:
        print("requirements.txt not found, installing basic packages...")
        basic_packages = ["torch", "torchvision", "numpy", "pandas", "matplotlib", "opencv-python", "tqdm", "psutil"]
        for package in basic_packages:
            if not run_command([sys.executable, "-m", "pip", "install", package], 
                              f"Installing {package}"):
                return False
    
    return True

def install_nerfstudio():
    """Install Nerfstudio"""
    print("\nInstalling Nerfstudio...")
    
    # Install nerfstudio
    if not run_command([sys.executable, "-m", "pip", "install", "nerfstudio"], 
                      "Installing Nerfstudio"):
        return False
    
    # Verify installation
    if not run_command(["ns-train", "--help"], "Verifying Nerfstudio installation", check=False):
        print("⚠️  Nerfstudio installation verification failed")
        return False
    
    return True

def setup_gaussian_splatting():
    """Clone and setup Gaussian Splatting repository"""
    print("\nSetting up Gaussian Splatting...")
    
    gs_dir = Path("gaussian_splatting")
    gs_repo_dir = gs_dir / "gaussian-splatting"
    
    # Create directory
    gs_dir.mkdir(exist_ok=True)
    
    # Clone repository if not exists
    if not gs_repo_dir.exists():
        if not run_command([
            "git", "clone", "https://github.com/graphdeco-inria/gaussian-splatting.git",
            str(gs_repo_dir)
        ], "Cloning Gaussian Splatting repository"):
            return False
    else:
        print("✅ Gaussian Splatting repository already exists")
    
    # Install submodules and dependencies
    original_cwd = os.getcwd()
    try:
        os.chdir(gs_repo_dir)
        
        # Initialize submodules
        if not run_command(["git", "submodule", "init"], "Initializing submodules"):
            return False
        
        if not run_command(["git", "submodule", "update"], "Updating submodules"):
            return False
        
        # Install CUDA extensions
        rasterization_dir = gs_repo_dir / "submodules" / "diff-gaussian-rasterization"
        if rasterization_dir.exists():
            os.chdir(rasterization_dir)
            if not run_command([sys.executable, "setup.py", "install"], 
                              "Installing differential gaussian rasterization"):
                print("⚠️  Failed to install CUDA extensions - may need manual setup")
        
        # Install simple-knn
        simple_knn_dir = gs_repo_dir / "submodules" / "simple-knn"
        if simple_knn_dir.exists():
            os.chdir(simple_knn_dir)
            if not run_command([sys.executable, "setup.py", "install"], 
                              "Installing simple-knn"):
                print("⚠️  Failed to install simple-knn - may need manual setup")
        
    finally:
        os.chdir(original_cwd)
    
    return True

def setup_colmap():
    """Setup COLMAP"""
    print("\nSetting up COLMAP...")
    
    # Check if COLMAP is already installed
    try:
        result = subprocess.run(["colmap", "--help"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ COLMAP already installed")
            return True
    except FileNotFoundError:
        pass
    
    print("COLMAP not found. Please install COLMAP manually:")
    print("Options:")
    print("1. Download pre-built binaries from: https://colmap.github.io/")
    print("2. Install via conda: conda install -c conda-forge colmap")
    print("3. Build from source: https://colmap.github.io/install.html")
    print("\nAfter installation, make sure 'colmap' is in your PATH")
    
    return False

def create_example_dataset():
    """Create example dataset structure"""
    print("\nCreating example dataset structure...")
    
    datasets_dir = Path("datasets")
    datasets_dir.mkdir(exist_ok=True)
    
    example_dir = datasets_dir / "example_scenes"
    example_dir.mkdir(exist_ok=True)
    
    # Create placeholder directories
    (example_dir / "indoor_scene_1").mkdir(exist_ok=True)
    (example_dir / "outdoor_scene_1").mkdir(exist_ok=True)
    
    # Create README
    readme_content = """# Example Datasets

To use this experiment pipeline, add your scene image directories here.

## Directory Structure
Each scene should be organized as:
```
datasets/example_scenes/
├── scene_name_1/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   ├── image_003.jpg
│   └── ... (more images)
└── scene_name_2/
    ├── image_001.jpg
    └── ... (more images)
```

## Image Requirements
- High-resolution images (recommended: 1K-2K pixels)
- Multiple viewpoints of the same scene
- Good overlap between consecutive images
- Sufficient lighting and sharp focus
- At least 20-50 images per scene for good reconstruction

## Scene Types
- Indoor scenes: rooms, offices, interior spaces
- Outdoor scenes: buildings, landmarks, natural environments
- Mixed scenes: combination of indoor/outdoor elements

## Data Capture Tips
1. Use consistent camera settings (fixed exposure if possible)
2. Maintain steady camera motion
3. Capture images in a systematic pattern (e.g., circular path)
4. Ensure good feature overlap between images
5. Avoid motion blur and poor lighting conditions
"""
    
    with open(example_dir / "README.md", "w") as f:
        f.write(readme_content)
    
    print("✅ Example dataset structure created")
    return True

def create_config_files():
    """Create configuration files"""
    print("\nCreating configuration files...")
    
    # Create a simple config file
    config_content = """# 3D Reconstruction Comparison Configuration

# Hardware Configuration
gpu_memory_limit: 8  # GB - adjust based on your GPU
num_parallel_jobs: 1  # Number of scenes to process in parallel

# Training Configuration
max_iterations: 30000
test_iterations: [7000, 15000, 30000]
train_test_split: 0.8  # 80% training, 20% testing

# Evaluation Configuration
inference_frames: 50  # Number of frames for timing inference
target_resolution: "1080p"

# Methods to compare
methods:
  - nerf_nerfacto
  - gaussian_splatting

# Output Configuration
save_rendered_images: true
save_timing_logs: true
generate_comparison_plots: true
"""
    
    with open("config.yaml", "w") as f:
        f.write(config_content)
    
    print("✅ Configuration files created")
    return True

def main():
    """Main setup function"""
    print("🚀 Setting up 3D Reconstruction Comparison Experiment")
    print("=" * 80)
    
    # Check system requirements
    if not check_system_requirements():
        print("❌ System requirements not met")
        return False
    
    # Install dependencies
    if not install_python_dependencies():
        print("❌ Failed to install Python dependencies")
        return False
    
    # Install Nerfstudio
    if not install_nerfstudio():
        print("❌ Failed to install Nerfstudio")
        return False
    
    # Setup Gaussian Splatting
    if not setup_gaussian_splatting():
        print("❌ Failed to setup Gaussian Splatting")
        return False
    
    # Setup COLMAP (informational)
    setup_colmap()
    
    # Create example dataset structure
    create_example_dataset()
    
    # Create config files
    create_config_files()
    
    print("\n" + "=" * 80)
    print("🎉 SETUP COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Install COLMAP if not already done")
    print("2. Add your scene image directories to datasets/example_scenes/")
    print("3. Run the experiment: python experiment_pipeline.py")
    print("\nFor help and documentation, see the README files created.")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
