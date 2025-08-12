"""
Results Analysis and Visualization for 3D Reconstruction Comparison
This script analyzes the experimental results and creates comparison plots
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json
import argparse

class ResultsAnalyzer:
    """Analyzes and visualizes experimental results"""
    
    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.plots_dir = self.results_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        # Load results
        self.results_df = None
        self.load_results()
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def load_results(self):
        """Load experimental results from CSV"""
        results_file = self.results_dir / "comparison_results.csv"
        if results_file.exists():
            self.results_df = pd.read_csv(results_file)
            print(f"Loaded {len(self.results_df)} results from {results_file}")
        else:
            print(f"Results file not found: {results_file}")
            return False
        return True
    
    def create_performance_comparison(self):
        """Create performance comparison plots"""
        if self.results_df is None:
            return
        
        # Set up the plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('3D Reconstruction Methods Comparison', fontsize=16, fontweight='bold')
        
        # Define metrics and their properties
        metrics = [
            ('psnr', 'PSNR (dB)', 'higher_better'),
            ('ssim', 'SSIM', 'higher_better'),
            ('lpips', 'LPIPS', 'lower_better'),
            ('training_time_seconds', 'Training Time (seconds)', 'lower_better'),
            ('model_size_mb', 'Model Size (MB)', 'lower_better'),
            ('inference_time_per_frame_seconds', 'Inference Time per Frame (seconds)', 'lower_better')
        ]
        
        for idx, (metric, title, direction) in enumerate(metrics):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            # Create box plot
            sns.boxplot(data=self.results_df, x='method', y=metric, ax=ax)
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('')
            
            # Rotate x-axis labels for better readability
            ax.tick_params(axis='x', rotation=45)
            
            # Add mean values as text
            for i, method in enumerate(self.results_df['method'].unique()):
                method_data = self.results_df[self.results_df['method'] == method][metric]
                if len(method_data) > 0:
                    mean_val = method_data.mean()
                    ax.text(i, mean_val, f'{mean_val:.3f}', 
                           horizontalalignment='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Performance comparison plot saved to {self.plots_dir / 'performance_comparison.png'}")
    
    def create_quality_vs_speed_plot(self):
        """Create quality vs speed scatter plot"""
        if self.results_df is None:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Quality vs Speed Trade-offs', fontsize=16, fontweight='bold')
        
        quality_metrics = ['psnr', 'ssim', 'lpips']
        quality_labels = ['PSNR (dB)', 'SSIM', 'LPIPS (lower is better)']
        
        for idx, (metric, label) in enumerate(zip(quality_metrics, quality_labels)):
            ax = axes[idx]
            
            # Create scatter plot
            for method in self.results_df['method'].unique():
                method_data = self.results_df[self.results_df['method'] == method]
                ax.scatter(method_data['inference_time_per_frame_seconds'], 
                          method_data[metric], 
                          label=method, 
                          s=100, 
                          alpha=0.7)
            
            ax.set_xlabel('Inference Time per Frame (seconds)')
            ax.set_ylabel(label)
            ax.set_title(f'{label} vs Inference Speed')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'quality_vs_speed.png', dpi=300, bbox_inches='tight')
        print(f"Quality vs speed plot saved to {self.plots_dir / 'quality_vs_speed.png'}")
    
    def create_training_analysis(self):
        """Create training time and memory analysis"""
        if self.results_df is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Training Analysis', fontsize=16, fontweight='bold')
        
        # Training time comparison
        ax1 = axes[0, 0]
        sns.barplot(data=self.results_df, x='method', y='training_time_seconds', ax=ax1)
        ax1.set_title('Training Time Comparison')
        ax1.set_ylabel('Training Time (seconds)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Memory usage comparison
        ax2 = axes[0, 1]
        sns.barplot(data=self.results_df, x='method', y='training_memory_gb', ax=ax2)
        ax2.set_title('Peak Memory Usage')
        ax2.set_ylabel('Memory Usage (GB)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Model size comparison
        ax3 = axes[1, 0]
        sns.barplot(data=self.results_df, x='method', y='model_size_mb', ax=ax3)
        ax3.set_title('Model Size Comparison')
        ax3.set_ylabel('Model Size (MB)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Training efficiency (quality per time)
        ax4 = axes[1, 1]
        # Calculate efficiency as PSNR per minute of training
        self.results_df['efficiency'] = self.results_df['psnr'] / (self.results_df['training_time_seconds'] / 60)
        sns.barplot(data=self.results_df, x='method', y='efficiency', ax=ax4)
        ax4.set_title('Training Efficiency (PSNR per minute)')
        ax4.set_ylabel('PSNR / Training Time (min)')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'training_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Training analysis plot saved to {self.plots_dir / 'training_analysis.png'}")
    
    def create_scene_comparison(self):
        """Create per-scene comparison"""
        if self.results_df is None:
            return
        
        scenes = self.results_df['scene_name'].unique()
        if len(scenes) < 2:
            print("Need at least 2 scenes for scene comparison")
            return
        
        # Quality metrics comparison across scenes
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Per-Scene Quality Comparison', fontsize=16, fontweight='bold')
        
        metrics = ['psnr', 'ssim', 'lpips', 'inference_time_per_frame_seconds']
        titles = ['PSNR', 'SSIM', 'LPIPS', 'Inference Time per Frame']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            # Pivot data for grouped bar plot
            pivot_data = self.results_df.pivot(index='scene_name', columns='method', values=metric)
            
            # Create grouped bar plot
            pivot_data.plot(kind='bar', ax=ax)
            ax.set_title(f'{title} by Scene')
            ax.set_ylabel(title)
            ax.set_xlabel('Scene')
            ax.legend(title='Method')
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'scene_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Scene comparison plot saved to {self.plots_dir / 'scene_comparison.png'}")
    
    def create_summary_report(self):
        """Create a summary report with statistics"""
        if self.results_df is None:
            return
        
        # Calculate summary statistics
        summary_stats = self.results_df.groupby('method').agg({
            'psnr': ['mean', 'std', 'min', 'max'],
            'ssim': ['mean', 'std', 'min', 'max'],
            'lpips': ['mean', 'std', 'min', 'max'],
            'training_time_seconds': ['mean', 'std', 'min', 'max'],
            'model_size_mb': ['mean', 'std', 'min', 'max'],
            'inference_time_per_frame_seconds': ['mean', 'std', 'min', 'max']
        }).round(4)
        
        # Flatten column names
        summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns]
        
        # Create summary report
        report_content = f"""
# 3D Reconstruction Methods Comparison Report

## Experiment Summary
- **Total Scenes Processed**: {len(self.results_df['scene_name'].unique())}
- **Methods Compared**: {', '.join(self.results_df['method'].unique())}
- **Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary Statistics

{summary_stats.to_string()}

## Key Findings

### Visual Quality (Higher is Better for PSNR/SSIM, Lower for LPIPS)
"""
        
        # Add specific comparisons
        methods = self.results_df['method'].unique()
        if len(methods) == 2:
            method1, method2 = methods
            
            psnr_diff = self.results_df[self.results_df['method'] == method1]['psnr'].mean() - \
                       self.results_df[self.results_df['method'] == method2]['psnr'].mean()
            
            ssim_diff = self.results_df[self.results_df['method'] == method1]['ssim'].mean() - \
                       self.results_df[self.results_df['method'] == method2]['ssim'].mean()
            
            lpips_diff = self.results_df[self.results_df['method'] == method1]['lpips'].mean() - \
                        self.results_df[self.results_df['method'] == method2]['lpips'].mean()
            
            time_diff = self.results_df[self.results_df['method'] == method1]['training_time_seconds'].mean() - \
                       self.results_df[self.results_df['method'] == method2]['training_time_seconds'].mean()
            
            inference_diff = self.results_df[self.results_df['method'] == method1]['inference_time_per_frame_seconds'].mean() - \
                           self.results_df[self.results_df['method'] == method2]['inference_time_per_frame_seconds'].mean()
            
            report_content += f"""
- **PSNR**: {method1} vs {method2} = {psnr_diff:+.2f} dB
- **SSIM**: {method1} vs {method2} = {ssim_diff:+.4f}
- **LPIPS**: {method1} vs {method2} = {lpips_diff:+.4f}

### Performance
- **Training Time**: {method1} vs {method2} = {time_diff:+.1f} seconds
- **Inference Speed**: {method1} vs {method2} = {inference_diff:+.4f} seconds per frame

### Recommendations
"""
            
            # Add automatic recommendations
            if psnr_diff > 1:
                report_content += f"- {method1} shows significantly better PSNR (+{psnr_diff:.1f} dB)\n"
            elif psnr_diff < -1:
                report_content += f"- {method2} shows significantly better PSNR (+{-psnr_diff:.1f} dB)\n"
            
            if inference_diff < -0.01:
                report_content += f"- {method1} is significantly faster for inference ({-inference_diff:.3f}s faster per frame)\n"
            elif inference_diff > 0.01:
                report_content += f"- {method2} is significantly faster for inference ({inference_diff:.3f}s faster per frame)\n"
        
        # Save report
        report_file = self.results_dir / 'experiment_report.md'
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        print(f"Summary report saved to {report_file}")
        return report_content
    
    def generate_all_plots(self):
        """Generate all analysis plots and reports"""
        print("Generating comprehensive analysis...")
        
        self.create_performance_comparison()
        self.create_quality_vs_speed_plot()
        self.create_training_analysis()
        self.create_scene_comparison()
        report = self.create_summary_report()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"Plots saved to: {self.plots_dir}")
        print(f"Report saved to: {self.results_dir / 'experiment_report.md'}")
        
        return report

def main():
    """Main function for results analysis"""
    parser = argparse.ArgumentParser(description='Analyze 3D reconstruction experiment results')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory containing experimental results')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save analysis outputs (default: same as results-dir)')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return
    
    # Initialize analyzer
    analyzer = ResultsAnalyzer(results_dir)
    
    # Generate analysis
    if analyzer.results_df is not None:
        analyzer.generate_all_plots()
    else:
        print("No results to analyze")

if __name__ == "__main__":
    main()
