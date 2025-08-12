"""
3D Reconstruction Comparison: Gaussian Splatting vs NeRF
Experiment Pipeline for comparing visual quality, processing time, and output file size

Based on methodology:
- Visual Quality: PSNR, SSIM, LPIPS on novel view synthesis
- Processing Time: Training time and inference time (1080p rendering)
- Output File Size: Model storage requirements

Requirements:
- Nerfstudio installed with nerfacto method
- 3D Gaussian Splatting repository cloned
- COLMAP for SfM preprocessing
- GPU with sufficient VRAM (RTX 3060 Mobile or better)
"""

import os
import sys
import json
import time
import csv
import shutil
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import torch
import psutil
from datetime import datetime

class ExperimentConfig:
    """Configuration class for the 3D reconstruction experiment"""
    
    def __init__(self):
        # Paths configuration
        self.base_dir = Path(__file__).parent
        self.data_dir = self.base_dir / "datasets"
        self.results_dir = self.base_dir / "results"
        self.colmap_dir = self.base_dir / "colmap_workspace"
        
        # Methods directories
        self.nerfstudio_dir = self.base_dir / "nerfstudio_outputs"
        self.gaussian_splatting_dir = self.base_dir / "gaussian_splatting"
        self.gs_repo_path = self.gaussian_splatting_dir / "gaussian-splatting"
        
        # Training parameters
        self.train_split_ratio = 0.8  # 80% training, 20% validation
        self.resolution = "1080p"  # Target resolution for inference timing
        self.gpu_memory_limit = 8  # GB, adjust based on RTX 3060 Mobile
        
        # Evaluation parameters
        self.test_iterations_nerf = [7000, 15000, 30000]
        self.test_iterations_gs = [7000, 30000]
        self.num_inference_frames = 50  # For timing inference
        
        # Output files
        self.results_csv = self.results_dir / "comparison_results.csv"
        self.detailed_results_csv = self.results_dir / "detailed_metrics.csv"
        self.timing_csv = self.results_dir / "timing_results.csv"
        
        # Create directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories for the experiment"""
        for directory in [self.data_dir, self.results_dir, self.colmap_dir, 
                         self.nerfstudio_dir, self.gaussian_splatting_dir]:
            directory.mkdir(parents=True, exist_ok=True)

class ColmapProcessor:
    """Handles COLMAP Structure-from-Motion preprocessing"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def process_scene(self, scene_path: Path, output_path: Path) -> bool:
        """
        Process a scene with COLMAP for SfM
        
        Args:
            scene_path: Path to input images
            output_path: Path to save COLMAP results
            
        Returns:
            bool: Success status
        """
        try:
            # Create COLMAP workspace
            colmap_workspace = output_path / "colmap"
            colmap_workspace.mkdir(parents=True, exist_ok=True)
            
            # COLMAP feature extraction
            self.logger.info(f"Running COLMAP feature extraction for {scene_path.name}")
            cmd_extract = [
                "colmap", "feature_extractor",
                "--database_path", str(colmap_workspace / "database.db"),
                "--image_path", str(scene_path),
                "--ImageReader.single_camera", "1"
            ]
            subprocess.run(cmd_extract, check=True, capture_output=True)
            
            # COLMAP feature matching
            self.logger.info(f"Running COLMAP feature matching for {scene_path.name}")
            cmd_match = [
                "colmap", "exhaustive_matcher",
                "--database_path", str(colmap_workspace / "database.db")
            ]
            subprocess.run(cmd_match, check=True, capture_output=True)
            
            # COLMAP sparse reconstruction
            self.logger.info(f"Running COLMAP sparse reconstruction for {scene_path.name}")
            sparse_dir = colmap_workspace / "sparse"
            sparse_dir.mkdir(parents=True, exist_ok=True)
            
            cmd_sparse = [
                "colmap", "mapper",
                "--database_path", str(colmap_workspace / "database.db"),
                "--image_path", str(scene_path),
                "--output_path", str(sparse_dir)
            ]
            subprocess.run(cmd_sparse, check=True, capture_output=True)
            
            self.logger.info(f"COLMAP processing completed for {scene_path.name}")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"COLMAP processing failed for {scene_path.name}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error in COLMAP processing: {e}")
            return False
    
    def convert_to_nerfstudio_format(self, colmap_path: Path, output_path: Path) -> bool:
        """Convert COLMAP output to nerfstudio format"""
        try:
            cmd = [
                "ns-process-data", "colmap",
                "--data", str(colmap_path.parent),
                "--output-dir", str(output_path)
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except Exception as e:
            self.logger.error(f"Failed to convert to nerfstudio format: {e}")
            return False

class NerfstudioTrainer:
    """Handles NeRF training using Nerfstudio"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def train_nerf(self, data_path: Path, output_path: Path, scene_name: str) -> Tuple[bool, Dict]:
        """
        Train NeRF using nerfacto method
        
        Args:
            data_path: Path to nerfstudio formatted data
            output_path: Path to save trained model
            scene_name: Name of the scene
            
        Returns:
            Tuple[bool, Dict]: Success status and metrics
        """
        metrics = {
            'scene_name': scene_name,
            'method': 'NeRF (nerfacto)',
            'training_time': 0,
            'peak_memory': 0,
            'model_size_mb': 0,
            'training_iterations': 30000
        }
        
        try:
            start_time = time.time()
            start_memory = self._get_gpu_memory()
            
            # Train NeRF with nerfacto
            self.logger.info(f"Starting NeRF training for {scene_name}")
            cmd = [
                "ns-train", "nerfacto",
                "--data", str(data_path),
                "--output-dir", str(output_path),
                "--max-num-iterations", "30000",
                "--eval-mode", "eval",
                "--viewer.quit-on-train-completion", "True"
            ]
            
            # Monitor memory during training
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            max_memory = start_memory
            
            while process.poll() is None:
                current_memory = self._get_gpu_memory()
                max_memory = max(max_memory, current_memory)
                time.sleep(10)  # Check every 10 seconds
            
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                self.logger.error(f"NeRF training failed: {stderr}")
                return False, metrics
            
            end_time = time.time()
            training_time = end_time - start_time
            
            # Calculate model size
            model_size = self._calculate_model_size(output_path)
            
            metrics.update({
                'training_time': training_time,
                'peak_memory': max_memory - start_memory,
                'model_size_mb': model_size,
                'success': True
            })
            
            self.logger.info(f"NeRF training completed for {scene_name} in {training_time:.2f} seconds")
            return True, metrics
            
        except Exception as e:
            self.logger.error(f"NeRF training failed for {scene_name}: {e}")
            metrics['success'] = False
            return False, metrics
    
    def evaluate_nerf(self, model_path: Path, scene_name: str) -> Dict:
        """Evaluate trained NeRF model"""
        try:
            # Run evaluation
            cmd = [
                "ns-eval",
                "--load-config", str(model_path / "config.yml"),
                "--output-path", str(model_path / "eval_results.json")
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Load evaluation results
            eval_file = model_path / "eval_results.json"
            if eval_file.exists():
                with open(eval_file, 'r') as f:
                    results = json.load(f)
                return results
            else:
                self.logger.warning(f"Evaluation results not found for {scene_name}")
                return {}
                
        except Exception as e:
            self.logger.error(f"NeRF evaluation failed for {scene_name}: {e}")
            return {}
    
    def measure_inference_time(self, model_path: Path, num_frames: int = 50) -> float:
        """Measure inference time for novel view synthesis"""
        try:
            start_time = time.time()
            
            # Render frames for timing
            cmd = [
                "ns-render", "interpolate",
                "--load-config", str(model_path / "config.yml"),
                "--output-path", str(model_path / "timing_renders"),
                "--frame-rate", "30",
                "--seconds", str(num_frames / 30)
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            
            end_time = time.time()
            total_time = end_time - start_time
            time_per_frame = total_time / num_frames
            
            return time_per_frame
            
        except Exception as e:
            self.logger.error(f"NeRF inference timing failed: {e}")
            return 0.0
    
    def _get_gpu_memory(self) -> float:
        """Get current GPU memory usage in GB"""
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024**3
            return 0.0
        except:
            return 0.0
    
    def _calculate_model_size(self, model_path: Path) -> float:
        """Calculate model size in MB"""
        try:
            total_size = 0
            for file_path in model_path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            return total_size / (1024 * 1024)  # Convert to MB
        except:
            return 0.0

class GaussianSplattingTrainer:
    """Handles Gaussian Splatting training using official implementation"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.gs_script_path = config.gs_repo_path / "train.py"
        self.gs_render_script = config.gs_repo_path / "render.py"
        self.gs_metrics_script = config.gs_repo_path / "metrics.py"
    
    def train_gaussian_splatting(self, data_path: Path, output_path: Path, scene_name: str) -> Tuple[bool, Dict]:
        """
        Train Gaussian Splatting model
        
        Args:
            data_path: Path to COLMAP data
            output_path: Path to save trained model
            scene_name: Name of the scene
            
        Returns:
            Tuple[bool, Dict]: Success status and metrics
        """
        metrics = {
            'scene_name': scene_name,
            'method': 'Gaussian Splatting',
            'training_time': 0,
            'peak_memory': 0,
            'model_size_mb': 0,
            'training_iterations': 30000
        }
        
        try:
            start_time = time.time()
            start_memory = self._get_gpu_memory()
            
            # Train Gaussian Splatting
            self.logger.info(f"Starting Gaussian Splatting training for {scene_name}")
            cmd = [
                "python", str(self.gs_script_path),
                "-s", str(data_path),
                "-m", str(output_path),
                "--eval",
                "--iterations", "30000",
                "--test_iterations", "7000", "30000",
                "--save_iterations", "7000", "30000"
            ]
            
            # Change to GS repository directory
            original_cwd = os.getcwd()
            os.chdir(self.config.gs_repo_path)
            
            # Monitor memory during training
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            max_memory = start_memory
            
            while process.poll() is None:
                current_memory = self._get_gpu_memory()
                max_memory = max(max_memory, current_memory)
                time.sleep(10)  # Check every 10 seconds
            
            stdout, stderr = process.communicate()
            os.chdir(original_cwd)
            
            if process.returncode != 0:
                self.logger.error(f"Gaussian Splatting training failed: {stderr}")
                return False, metrics
            
            end_time = time.time()
            training_time = end_time - start_time
            
            # Calculate model size
            model_size = self._calculate_model_size(output_path)
            
            metrics.update({
                'training_time': training_time,
                'peak_memory': max_memory - start_memory,
                'model_size_mb': model_size,
                'success': True
            })
            
            self.logger.info(f"Gaussian Splatting training completed for {scene_name} in {training_time:.2f} seconds")
            return True, metrics
            
        except Exception as e:
            self.logger.error(f"Gaussian Splatting training failed for {scene_name}: {e}")
            metrics['success'] = False
            return False, metrics
    
    def evaluate_gaussian_splatting(self, model_path: Path, data_path: Path, scene_name: str) -> Dict:
        """Evaluate trained Gaussian Splatting model"""
        try:
            # Render test images
            original_cwd = os.getcwd()
            os.chdir(self.config.gs_repo_path)
            
            self.logger.info(f"Rendering test images for {scene_name}")
            render_cmd = [
                "python", str(self.gs_render_script),
                "-m", str(model_path),
                "-s", str(data_path)
            ]
            subprocess.run(render_cmd, check=True, capture_output=True)
            
            # Calculate metrics
            self.logger.info(f"Calculating metrics for {scene_name}")
            metrics_cmd = [
                "python", str(self.gs_metrics_script),
                "-m", str(model_path)
            ]
            subprocess.run(metrics_cmd, check=True, capture_output=True)
            
            os.chdir(original_cwd)
            
            # Load results
            results_file = model_path / "results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                return results
            else:
                self.logger.warning(f"Results file not found for {scene_name}")
                return {}
                
        except Exception as e:
            self.logger.error(f"Gaussian Splatting evaluation failed for {scene_name}: {e}")
            return {}
    
    def measure_inference_time(self, model_path: Path, data_path: Path, num_frames: int = 50) -> float:
        """Measure inference time for rendering"""
        try:
            start_time = time.time()
            
            # Render frames for timing
            original_cwd = os.getcwd()
            os.chdir(self.config.gs_repo_path)
            
            cmd = [
                "python", str(self.gs_render_script),
                "-m", str(model_path),
                "-s", str(data_path),
                "--skip_train"
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            
            os.chdir(original_cwd)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Count rendered frames
            test_dir = model_path / "test"
            if test_dir.exists():
                renders_dir = test_dir / "ours_30000" / "renders"
                if renders_dir.exists():
                    num_rendered = len(list(renders_dir.glob("*.png")))
                    if num_rendered > 0:
                        return total_time / num_rendered
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Gaussian Splatting inference timing failed: {e}")
            return 0.0
    
    def _get_gpu_memory(self) -> float:
        """Get current GPU memory usage in GB"""
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024**3
            return 0.0
        except:
            return 0.0
    
    def _calculate_model_size(self, model_path: Path) -> float:
        """Calculate model size in MB"""
        try:
            total_size = 0
            for file_path in model_path.rglob("*.ply"):  # Gaussian splatting models are .ply files
                total_size += file_path.stat().st_size
            for file_path in model_path.rglob("*.pth"):  # PyTorch model files
                total_size += file_path.stat().st_size
            return total_size / (1024 * 1024)  # Convert to MB
        except:
            return 0.0

class ExperimentRunner:
    """Main experiment runner that coordinates the entire pipeline"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Initialize components
        self.colmap_processor = ColmapProcessor(config)
        self.nerf_trainer = NerfstudioTrainer(config)
        self.gs_trainer = GaussianSplattingTrainer(config)
        
        # Results storage
        self.all_results = []
        self.detailed_metrics = []
        self.timing_results = []
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.results_dir / 'experiment.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        return logging.getLogger(__name__)
    
    def run_full_experiment(self, scene_directories: List[Path]) -> None:
        """
        Run the complete experiment pipeline on multiple scenes
        
        Args:
            scene_directories: List of paths to scene image directories
        """
        self.logger.info("Starting 3D Reconstruction Comparison Experiment")
        self.logger.info(f"GPU Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU Only'}")
        self.logger.info(f"Processing {len(scene_directories)} scenes")
        
        for scene_dir in scene_directories:
            scene_name = scene_dir.name
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Processing Scene: {scene_name}")
            self.logger.info(f"{'='*60}")
            
            # Process scene with both methods
            self._process_single_scene(scene_dir, scene_name)
        
        # Save all results
        self._save_results()
        self.logger.info("Experiment completed successfully!")
    
    def _process_single_scene(self, scene_dir: Path, scene_name: str) -> None:
        """Process a single scene with both NeRF and Gaussian Splatting"""
        
        # Step 1: COLMAP preprocessing
        self.logger.info(f"Step 1: COLMAP preprocessing for {scene_name}")
        colmap_output = self.config.colmap_dir / scene_name
        
        if not self.colmap_processor.process_scene(scene_dir, colmap_output):
            self.logger.error(f"COLMAP preprocessing failed for {scene_name}")
            return
        
        # Step 2: NeRF Training and Evaluation
        self.logger.info(f"Step 2: NeRF training and evaluation for {scene_name}")
        self._run_nerf_pipeline(colmap_output, scene_name)
        
        # Step 3: Gaussian Splatting Training and Evaluation
        self.logger.info(f"Step 3: Gaussian Splatting training and evaluation for {scene_name}")
        self._run_gaussian_splatting_pipeline(colmap_output, scene_name)
    
    def _run_nerf_pipeline(self, colmap_path: Path, scene_name: str) -> None:
        """Run complete NeRF pipeline"""
        try:
            # Convert to nerfstudio format
            nerf_data_path = self.config.nerfstudio_dir / f"{scene_name}_data"
            if not self.colmap_processor.convert_to_nerfstudio_format(colmap_path / "colmap", nerf_data_path):
                self.logger.error(f"Failed to convert {scene_name} to nerfstudio format")
                return
            
            # Train NeRF
            nerf_output_path = self.config.nerfstudio_dir / f"{scene_name}_nerf"
            success, training_metrics = self.nerf_trainer.train_nerf(nerf_data_path, nerf_output_path, scene_name)
            
            if not success:
                self.logger.error(f"NeRF training failed for {scene_name}")
                return
            
            # Find the actual model directory (nerfstudio creates timestamped directories)
            model_dirs = list(nerf_output_path.glob("nerfacto/*/"))
            if not model_dirs:
                self.logger.error(f"No NeRF model found for {scene_name}")
                return
            
            model_path = model_dirs[0]  # Use the most recent
            
            # Evaluate NeRF
            eval_metrics = self.nerf_trainer.evaluate_nerf(model_path, scene_name)
            
            # Measure inference time
            inference_time = self.nerf_trainer.measure_inference_time(model_path, self.config.num_inference_frames)
            
            # Store results
            result = {
                'scene_name': scene_name,
                'method': 'NeRF (nerfacto)',
                'training_time_seconds': training_metrics.get('training_time', 0),
                'training_memory_gb': training_metrics.get('peak_memory', 0),
                'model_size_mb': training_metrics.get('model_size_mb', 0),
                'inference_time_per_frame_seconds': inference_time,
                'psnr': eval_metrics.get('psnr', 0),
                'ssim': eval_metrics.get('ssim', 0),
                'lpips': eval_metrics.get('lpips', 0),
                'success': True
            }
            
            self.all_results.append(result)
            self.logger.info(f"NeRF pipeline completed successfully for {scene_name}")
            
        except Exception as e:
            self.logger.error(f"NeRF pipeline failed for {scene_name}: {e}")
    
    def _run_gaussian_splatting_pipeline(self, colmap_path: Path, scene_name: str) -> None:
        """Run complete Gaussian Splatting pipeline"""
        try:
            # Use COLMAP data directly
            gs_data_path = colmap_path / "colmap"
            
            # Train Gaussian Splatting
            gs_output_path = self.config.gaussian_splatting_dir / f"{scene_name}_gs"
            success, training_metrics = self.gs_trainer.train_gaussian_splatting(gs_data_path, gs_output_path, scene_name)
            
            if not success:
                self.logger.error(f"Gaussian Splatting training failed for {scene_name}")
                return
            
            # Evaluate Gaussian Splatting
            eval_metrics = self.gs_trainer.evaluate_gaussian_splatting(gs_output_path, gs_data_path, scene_name)
            
            # Measure inference time
            inference_time = self.gs_trainer.measure_inference_time(gs_output_path, gs_data_path, self.config.num_inference_frames)
            
            # Extract metrics from results
            psnr = ssim = lpips = 0
            if eval_metrics:
                # Gaussian Splatting metrics.py creates results with method keys
                for method_key, method_results in eval_metrics.items():
                    if isinstance(method_results, dict):
                        psnr = method_results.get('PSNR', 0)
                        ssim = method_results.get('SSIM', 0)
                        lpips = method_results.get('LPIPS', 0)
                        break
            
            # Store results
            result = {
                'scene_name': scene_name,
                'method': 'Gaussian Splatting',
                'training_time_seconds': training_metrics.get('training_time', 0),
                'training_memory_gb': training_metrics.get('peak_memory', 0),
                'model_size_mb': training_metrics.get('model_size_mb', 0),
                'inference_time_per_frame_seconds': inference_time,
                'psnr': psnr,
                'ssim': ssim,
                'lpips': lpips,
                'success': True
            }
            
            self.all_results.append(result)
            self.logger.info(f"Gaussian Splatting pipeline completed successfully for {scene_name}")
            
        except Exception as e:
            self.logger.error(f"Gaussian Splatting pipeline failed for {scene_name}: {e}")
    
    def _save_results(self) -> None:
        """Save all experimental results to CSV files"""
        
        # Main results comparison
        if self.all_results:
            df_results = pd.DataFrame(self.all_results)
            df_results.to_csv(self.config.results_csv, index=False)
            self.logger.info(f"Results saved to {self.config.results_csv}")
            
            # Print summary
            print("\n" + "="*80)
            print("EXPERIMENT RESULTS SUMMARY")
            print("="*80)
            print(df_results.to_string(index=False))
            print("="*80)
        
        # Save detailed metrics if available
        if self.detailed_metrics:
            df_detailed = pd.DataFrame(self.detailed_metrics)
            df_detailed.to_csv(self.config.detailed_results_csv, index=False)
            self.logger.info(f"Detailed metrics saved to {self.config.detailed_results_csv}")
        
        # Create summary statistics
        self._create_summary_statistics()
    
    def _create_summary_statistics(self) -> None:
        """Create summary statistics comparing both methods"""
        if not self.all_results:
            return
        
        df = pd.DataFrame(self.all_results)
        
        # Group by method
        summary_stats = df.groupby('method').agg({
            'training_time_seconds': ['mean', 'std'],
            'training_memory_gb': ['mean', 'std'],
            'model_size_mb': ['mean', 'std'],
            'inference_time_per_frame_seconds': ['mean', 'std'],
            'psnr': ['mean', 'std'],
            'ssim': ['mean', 'std'],
            'lpips': ['mean', 'std']
        }).round(4)
        
        # Flatten column names
        summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns]
        
        # Save summary
        summary_file = self.config.results_dir / "summary_statistics.csv"
        summary_stats.to_csv(summary_file)
        
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        print(summary_stats.to_string())
        print("="*80)

def main():
    """Main function to run the experiment"""
    
    # Create configuration
    config = ExperimentConfig()
    
    # Initialize experiment runner
    runner = ExperimentRunner(config)
    
    # Example scene directories (replace with your actual scene paths)
    scene_directories = [
        # Add your scene directories here, e.g.:
        # Path("path/to/your/indoor_scene_1"),
        # Path("path/to/your/outdoor_scene_1"),
        # Path("path/to/your/indoor_scene_2"),
    ]
    
    # Check for example data directory
    example_scenes_dir = config.data_dir / "example_scenes"
    if example_scenes_dir.exists():
        scene_directories.extend([d for d in example_scenes_dir.iterdir() if d.is_dir()])
    
    if not scene_directories:
        print("No scene directories found!")
        print("Please add your scene image directories to the experiment.")
        print("Example directory structure:")
        print("datasets/")
        print("├── indoor_scene_1/")
        print("│   ├── IMG_001.jpg")
        print("│   ├── IMG_002.jpg")
        print("│   └── ...")
        print("└── outdoor_scene_1/")
        print("    ├── IMG_001.jpg")
        print("    ├── IMG_002.jpg")
        print("    └── ...")
        return
    
    # Run the experiment
    try:
        runner.run_full_experiment(scene_directories)
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
    except Exception as e:
        print(f"Experiment failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
