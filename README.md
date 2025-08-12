# 3D Reconstruction Comparison: Gaussian Splatting vs NeRF

This repository contains a comprehensive experiment pipeline for comparing **3D Gaussian Splatting** and **Neural Radiance Fields (NeRF)** as 3D reconstruction methods. The comparison focuses on three main criteria: **visual quality**, **processing time**, and **output file size**.

## 📋 Overview

The experiment pipeline implements the methodology described in the research proposal:

- **Visual Quality Evaluation**: PSNR, SSIM, and LPIPS metrics on novel view synthesis
- **Processing Time Analysis**: Training time and inference time (1080p rendering)  
- **Storage Analysis**: Model file sizes and memory requirements
- **Automated Pipeline**: End-to-end processing from raw images to comparative results

## 🏗️ Architecture

### Methods Compared

1. **NeRF (Neural Radiance Fields)**
   - Implementation: Nerfstudio's "nerfacto" method
   - Based on Instant Neural Graphics Primitives (Instant-NGP)
   - Optimized for training speed and high-fidelity results

2. **3D Gaussian Splatting**
   - Implementation: Official repository from the original paper
   - Real-time rendering capabilities
   - High photorealism with efficient training

### Pipeline Components

```
Raw Images → COLMAP (SfM) → NeRF Training → Evaluation → Results
                         ↘ GS Training → Evaluation → Comparison
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone this repository
git clone <repository-url>
cd 3d-reconstruction-comparison

# Run setup script (installs dependencies and repositories)
python setup_environment.py
```

### 2. Manual Installation (if setup script fails)

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Nerfstudio
pip install nerfstudio

# Install COLMAP (choose one method):
# Option 1: Conda
conda install -c conda-forge colmap

# Option 2: Download binaries from https://colmap.github.io/

# Clone Gaussian Splatting repository
git clone https://github.com/graphdeco-inria/gaussian-splatting.git gaussian_splatting/gaussian-splatting
cd gaussian_splatting/gaussian-splatting
git submodule init
git submodule update

# Install CUDA extensions (requires CUDA toolkit)
cd submodules/diff-gaussian-rasterization
python setup.py install
cd ../simple-knn  
python setup.py install
```

### 3. Prepare Your Data

```bash
# Create dataset directory structure
mkdir -p datasets/example_scenes

# Add your scene directories:
datasets/example_scenes/
├── indoor_scene_1/
│   ├── IMG_001.jpg
│   ├── IMG_002.jpg
│   └── ... (20-50+ images)
└── outdoor_scene_1/
    ├── IMG_001.jpg
    └── ... (20-50+ images)
```

### 4. Run Experiment

```bash
# Run the complete experiment pipeline
python experiment_pipeline.py
```

### 5. Analyze Results

```bash
# Generate analysis plots and reports
python analyze_results.py --results-dir results
```

## 📊 Output and Results

The experiment generates comprehensive results including:

### CSV Files
- `comparison_results.csv` - Main comparison metrics
- `detailed_metrics.csv` - Per-image detailed metrics  
- `timing_results.csv` - Detailed timing information
- `summary_statistics.csv` - Statistical summary

### Visualizations
- Performance comparison box plots
- Quality vs Speed scatter plots
- Training analysis charts
- Per-scene comparisons
- Summary report in Markdown

### Example Results Structure
```
results/
├── comparison_results.csv
├── detailed_metrics.csv
├── summary_statistics.csv
├── experiment_report.md
├── experiment.log
└── plots/
    ├── performance_comparison.png
    ├── quality_vs_speed.png
    ├── training_analysis.png
    └── scene_comparison.png
```

## 🔧 Configuration

Edit `config.yaml` to customize experiment parameters:

```yaml
# Hardware Configuration
gpu_memory_limit: 8  # GB
num_parallel_jobs: 1

# Training Configuration  
max_iterations: 30000
test_iterations: [7000, 15000, 30000]
train_test_split: 0.8

# Evaluation Configuration
inference_frames: 50
target_resolution: "1080p"
```

## 📈 Metrics Explained

### Visual Quality Metrics

- **PSNR (Peak Signal-to-Noise Ratio)**: Higher is better (dB)
- **SSIM (Structural Similarity Index)**: Higher is better (0-1)  
- **LPIPS (Learned Perceptual Image Patch Similarity)**: Lower is better

### Performance Metrics

- **Training Time**: Total optimization time (seconds)
- **Peak Memory**: Maximum GPU memory usage during training (GB)
- **Model Size**: Storage requirements for trained model (MB)
- **Inference Time**: Time to render single 1080p frame (seconds)

## 🎯 Expected Results

Based on the methodology and existing research:

| Metric | NeRF (Nerfacto) | Gaussian Splatting | Winner |
|--------|-----------------|-------------------|---------|
| **PSNR** | High quality | High quality | Similar |
| **SSIM** | Good structure | Excellent structure | GS |
| **LPIPS** | Perceptually good | Perceptually excellent | GS |
| **Training Time** | Moderate | Fast | GS |
| **Inference Speed** | Slow | Real-time | **GS** |
| **Model Size** | Compact | Larger | NeRF |

## 🔍 Data Requirements

### Scene Capture Guidelines

1. **Image Count**: 20-100 images per scene
2. **Resolution**: 1K-2K pixels recommended
3. **Overlap**: 70%+ overlap between consecutive images
4. **Lighting**: Consistent, avoid extreme shadows/highlights
5. **Motion**: Steady camera movement, avoid blur

### Recommended Scenes

- **Indoor**: Rooms, offices, interior spaces
- **Outdoor**: Buildings, landmarks, natural environments  
- **Mixed**: Indoor/outdoor combinations

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce GPU memory usage
   # Edit config: gpu_memory_limit: 4
   # Or train smaller scenes first
   ```

2. **COLMAP Processing Fails**
   ```bash
   # Check image quality and overlap
   # Ensure sufficient texture/features
   # Try different COLMAP parameters
   ```

3. **Nerfstudio Installation Issues**
   ```bash
   # Use conda environment
   conda create -n nerf python=3.8
   conda activate nerf
   pip install nerfstudio
   ```

4. **Gaussian Splatting CUDA Extensions**
   ```bash
   # Ensure CUDA toolkit is installed
   # Check PyTorch CUDA version compatibility
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

### Performance Optimization

- **GPU Memory**: Use `--data_device cpu` for large datasets
- **Training Speed**: Reduce iterations for quick tests
- **Disk Space**: Expect 5-20GB per scene for outputs

## 📚 Technical Details

### Dependencies

- **Python**: 3.8+
- **PyTorch**: 2.0+ with CUDA support
- **Nerfstudio**: Latest version
- **COLMAP**: 3.7+
- **GPU**: NVIDIA with 8GB+ VRAM recommended

### Implementation Notes

- COLMAP performs Structure-from-Motion preprocessing
- Nerfstudio handles NeRF training with nerfacto method
- Official Gaussian Splatting repository for GS training
- Automated metric calculation using both frameworks
- Results aggregation and statistical analysis

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📄 License

This project follows the licenses of its components:
- Nerfstudio: Apache 2.0 License
- Gaussian Splatting: Custom Research License
- This pipeline: MIT License

## 🔗 References

- [Nerfstudio Documentation](https://docs.nerf.studio/)
- [COLMAP Documentation](https://colmap.github.io/)
- [3D Gaussian Splatting Paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [Original Gaussian Splatting Repository](https://github.com/graphdeco-inria/gaussian-splatting)

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review component documentation
3. Open an issue with detailed error logs
4. Include system specifications and data characteristics

---

**Note**: This experiment requires significant computational resources (GPU with 8GB+ VRAM) and may take several hours to complete depending on the number of scenes and their complexity.
