"""
Demo script for 3D Reconstruction Comparison Pipeline
This script demonstrates how to use the experiment pipeline with example data
"""

import sys
import time
from pathlib import Path
import shutil

def create_demo_data():
    """Create demo data structure with instructions"""
    print("Creating demo data structure...")
    
    base_dir = Path(__file__).parent
    datasets_dir = base_dir / "datasets" / "example_scenes"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    
    # Create example scene directories
    demo_scenes = ["living_room", "office_space", "garden_view"]
    
    for scene in demo_scenes:
        scene_dir = datasets_dir / scene
        scene_dir.mkdir(exist_ok=True)
        
        # Create instruction file
        instruction_file = scene_dir / "INSTRUCTIONS.txt"
        with open(instruction_file, "w") as f:
            f.write(f"""
DEMO SCENE: {scene.upper()}

To use this demo scene:

1. Add 20-100 high-resolution images of your scene to this directory
2. Images should be in JPG or PNG format
3. Ensure good overlap between consecutive images (70%+)
4. Use consistent lighting and avoid motion blur

Example file structure:
{scene}/
├── IMG_001.jpg
├── IMG_002.jpg
├── IMG_003.jpg
└── ... (more images)

Scene recommendations for '{scene}':
""")
            
            if "living_room" in scene:
                f.write("""
- Capture from multiple heights and angles
- Include furniture and wall details
- Walk around the room in a systematic pattern
- Ensure good lighting (avoid strong shadows)
""")
            elif "office" in scene:
                f.write("""
- Focus on desk area and office equipment  
- Capture wall decorations and bookshelves
- Include different viewing angles of workspace
- Ensure computer screens are not too bright
""")
            elif "garden" in scene:
                f.write("""
- Capture plants and outdoor features
- Use consistent lighting (avoid harsh shadows)
- Include different perspectives of landscape
- Consider wind effects on plants
""")
    
    print(f"✅ Demo structure created in {datasets_dir}")
    return datasets_dir

def run_demo():
    """Run a demonstration of the pipeline"""
    print("🎬 3D Reconstruction Comparison Demo")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("experiment_pipeline.py").exists():
        print("❌ Please run this script from the project root directory")
        return False
    
    # Create demo data
    datasets_dir = create_demo_data()
    
    # Check for actual image data
    scene_dirs = [d for d in datasets_dir.iterdir() if d.is_dir()]
    scenes_with_images = []
    
    for scene_dir in scene_dirs:
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            image_files.extend(list(scene_dir.glob(f"*{ext}")))
        
        if len(image_files) >= 10:  # Need at least 10 images
            scenes_with_images.append(scene_dir)
            print(f"✅ Found {len(image_files)} images in {scene_dir.name}")
        else:
            print(f"⚠️  {scene_dir.name} needs more images ({len(image_files)} found, need 10+)")
    
    if not scenes_with_images:
        print("\n📋 TO RUN THE DEMO:")
        print("1. Add images to the created scene directories")
        print("2. Run this demo script again")
        print("3. Or run the full pipeline: python experiment_pipeline.py")
        return False
    
    # Run the experiment
    print(f"\n🚀 Running experiment on {len(scenes_with_images)} scenes...")
    
    try:
        # Import and run the experiment
        from experiment_pipeline import ExperimentRunner, ExperimentConfig
        
        # Create config
        config = ExperimentConfig()
        
        # Initialize runner
        runner = ExperimentRunner(config)
        
        # Run experiment on scenes with images
        runner.run_full_experiment(scenes_with_images)
        
        # Analyze results
        print("\n📊 Analyzing results...")
        from analyze_results import ResultsAnalyzer
        
        analyzer = ResultsAnalyzer(config.results_dir)
        analyzer.generate_all_plots()
        
        print("\n🎉 Demo completed successfully!")
        print(f"📁 Results saved to: {config.results_dir}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all dependencies are installed")
        return False
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False

def show_usage():
    """Show usage instructions"""
    print("""
🎯 3D Reconstruction Comparison Pipeline Demo

USAGE:
    python demo.py                 # Run demo with example data
    python experiment_pipeline.py  # Run full experiment
    python analyze_results.py      # Analyze existing results

SETUP STEPS:
1. Install dependencies: python setup_environment.py
2. Add image data to datasets/example_scenes/
3. Run demo: python demo.py

REQUIREMENTS:
- Python 3.8+
- CUDA-capable GPU (8GB+ VRAM recommended)
- COLMAP installed
- Nerfstudio installed
- 20+ high-quality images per scene

For more information, see README.md
""")

def main():
    """Main demo function"""
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        show_usage()
        return
    
    print("Starting 3D Reconstruction Comparison Demo...")
    start_time = time.time()
    
    success = run_demo()
    
    end_time = time.time()
    duration = end_time - start_time
    
    if success:
        print(f"\n✅ Demo completed in {duration:.1f} seconds")
    else:
        print(f"\n❌ Demo setup completed in {duration:.1f} seconds")
        print("Add image data and run again to execute the full pipeline")

if __name__ == "__main__":
    main()
