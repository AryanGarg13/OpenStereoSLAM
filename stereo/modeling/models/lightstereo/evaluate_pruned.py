"""
Evaluation utilities for pruned LightStereo models.
Provides comprehensive evaluation metrics and performance benchmarking.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import logging

from .advanced_pruned_lightstereo import AdvancedPrunedLightStereo
from .lightstereo import LightStereo
from .pruning_utils import ModelAnalyzer


class StereoMetrics:
    """Stereo depth estimation evaluation metrics."""
    
    @staticmethod
    def calculate_disparity_metrics(pred: torch.Tensor, 
                                  gt: torch.Tensor, 
                                  max_disp: float,
                                  mask: torch.Tensor = None) -> Dict[str, float]:
        """
        Calculate stereo disparity metrics.
        
        Args:
            pred: Predicted disparity [B, H, W]
            gt: Ground truth disparity [B, H, W]
            max_disp: Maximum disparity value
            mask: Valid pixel mask [B, H, W]
        
        Returns:
            Dictionary of metrics
        """
        if mask is None:
            mask = (gt > 0) & (gt < max_disp)
        
        if mask.sum() == 0:
            return {
                'mae': float('inf'),
                'rmse': float('inf'),
                'bad_1': 1.0,
                'bad_2': 1.0,
                'bad_3': 1.0,
                'bad_5': 1.0
            }
        
        # Mean Absolute Error
        mae = torch.abs(pred[mask] - gt[mask]).mean().item()
        
        # Root Mean Square Error
        rmse = torch.sqrt(((pred[mask] - gt[mask]) ** 2).mean()).item()
        
        # Bad pixel ratios (error > threshold)
        error = torch.abs(pred[mask] - gt[mask])
        bad_1 = (error > 1.0).float().mean().item()
        bad_2 = (error > 2.0).float().mean().item()
        bad_3 = (error > 3.0).float().mean().item()
        bad_5 = (error > 5.0).float().mean().item()
        
        return {
            'mae': mae,
            'rmse': rmse,
            'bad_1': bad_1,
            'bad_2': bad_2,
            'bad_3': bad_3,
            'bad_5': bad_5
        }
    
    @staticmethod
    def calculate_depth_metrics(pred: torch.Tensor,
                              gt: torch.Tensor,
                              mask: torch.Tensor = None) -> Dict[str, float]:
        """Calculate depth estimation metrics."""
        if mask is None:
            mask = gt > 0
        
        if mask.sum() == 0:
            return {'depth_mae': float('inf'), 'depth_rmse': float('inf')}
        
        # Convert disparity to depth (assuming baseline and focal length)
        # This would need calibration parameters in practice
        pred_depth = 1.0 / (pred[mask] + 1e-6)
        gt_depth = 1.0 / (gt[mask] + 1e-6)
        
        depth_mae = torch.abs(pred_depth - gt_depth).mean().item()
        depth_rmse = torch.sqrt(((pred_depth - gt_depth) ** 2).mean()).item()
        
        return {
            'depth_mae': depth_mae,
            'depth_rmse': depth_rmse
        }


class PerformanceBenchmark:
    """Performance benchmarking utilities."""
    
    def __init__(self, device: torch.device):
        self.device = device
        
    def benchmark_inference_time(self, 
                                model: nn.Module,
                                input_shape: Tuple[int, int, int, int] = (1, 3, 480, 640),
                                num_runs: int = 100,
                                warmup_runs: int = 10) -> Dict[str, float]:
        """
        Benchmark model inference time.
        
        Args:
            model: Model to benchmark
            input_shape: Input tensor shape (B, C, H, W)
            num_runs: Number of inference runs
            warmup_runs: Number of warmup runs
        
        Returns:
            Timing statistics
        """
        model.eval()
        
        # Create dummy input
        dummy_left = torch.randn(input_shape, device=self.device)
        dummy_right = torch.randn(input_shape, device=self.device)
        dummy_data = {'left': dummy_left, 'right': dummy_right}
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(dummy_data)
        
        # Synchronize GPU
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(dummy_data)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.time()
                times.append(end_time - start_time)
        
        times = np.array(times)
        
        return {
            'mean_time': float(np.mean(times)),
            'std_time': float(np.std(times)),
            'min_time': float(np.min(times)),
            'max_time': float(np.max(times)),
            'fps': 1.0 / np.mean(times)
        }
    
    def benchmark_memory_usage(self, 
                             model: nn.Module,
                             input_shape: Tuple[int, int, int, int] = (1, 3, 480, 640)) -> Dict[str, float]:
        """
        Benchmark GPU memory usage.
        
        Args:
            model: Model to benchmark
            input_shape: Input tensor shape
        
        Returns:
            Memory usage statistics
        """
        if self.device.type != 'cuda':
            return {'memory_allocated': 0, 'memory_cached': 0}
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        model.eval()
        dummy_left = torch.randn(input_shape, device=self.device)
        dummy_right = torch.randn(input_shape, device=self.device)
        dummy_data = {'left': dummy_left, 'right': dummy_right}
        
        # Forward pass
        with torch.no_grad():
            _ = model(dummy_data)
        
        memory_allocated = torch.cuda.max_memory_allocated() / (1024**2)  # MB
        memory_cached = torch.cuda.max_memory_reserved() / (1024**2)  # MB
        
        return {
            'memory_allocated': memory_allocated,
            'memory_cached': memory_cached
        }


class PrunedModelEvaluator:
    """Comprehensive evaluator for pruned LightStereo models."""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.benchmark = PerformanceBenchmark(device)
        self.logger = logging.getLogger(__name__)
    
    def evaluate_model(self,
                      model: nn.Module,
                      dataloader: DataLoader,
                      model_name: str = "Model") -> Dict:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Model to evaluate
            dataloader: Evaluation data loader
            model_name: Name for logging
        
        Returns:
            Evaluation results
        """
        self.logger.info(f"Evaluating {model_name}...")
        
        model.eval()
        all_metrics = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data in dataloader:
                # Move data to device
                for key in batch_data:
                    if isinstance(batch_data[key], torch.Tensor):
                        batch_data[key] = batch_data[key].to(self.device)
                
                # Forward pass
                model_pred = model(batch_data)
                
                # Calculate loss
                if hasattr(model, 'get_loss'):
                    loss, _ = model.get_loss(model_pred, batch_data)
                    total_loss += loss.item()
                
                # Calculate stereo metrics
                if 'disp' in batch_data and 'disp_pred' in model_pred:
                    disp_pred = model_pred['disp_pred'].squeeze(1)  # Remove channel dim
                    disp_gt = batch_data['disp']
                    max_disp = getattr(model, 'max_disp', 192)
                    
                    metrics = StereoMetrics.calculate_disparity_metrics(
                        disp_pred, disp_gt, max_disp
                    )
                    all_metrics.append(metrics)
                
                num_batches += 1
        
        # Aggregate metrics
        if all_metrics:
            avg_metrics = {}
            for key in all_metrics[0].keys():
                avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        else:
            avg_metrics = {}
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'stereo_metrics': avg_metrics,
            'num_samples': num_batches * dataloader.batch_size
        }
    
    def compare_models(self,
                      original_model: LightStereo,
                      pruned_model: AdvancedPrunedLightStereo,
                      dataloader: DataLoader,
                      input_shape: Tuple[int, int, int, int] = (1, 3, 480, 640)) -> Dict:
        """
        Compare original and pruned models.
        
        Args:
            original_model: Original LightStereo model
            pruned_model: Pruned LightStereo model
            dataloader: Evaluation data loader
            input_shape: Input shape for benchmarking
        
        Returns:
            Comparison results
        """
        self.logger.info("Comparing original and pruned models...")
        
        # Move models to device
        original_model = original_model.to(self.device)
        pruned_model = pruned_model.to(self.device)
        
        # Evaluate accuracy
        original_results = self.evaluate_model(original_model, dataloader, "Original")
        pruned_results = self.evaluate_model(pruned_model, dataloader, "Pruned")
        
        # Benchmark performance
        original_timing = self.benchmark.benchmark_inference_time(original_model, input_shape)
        pruned_timing = self.benchmark.benchmark_inference_time(pruned_model, input_shape)
        
        original_memory = self.benchmark.benchmark_memory_usage(original_model, input_shape)
        pruned_memory = self.benchmark.benchmark_memory_usage(pruned_model, input_shape)
        
        # Model statistics
        original_params = ModelAnalyzer.count_parameters(original_model)
        pruned_params = ModelAnalyzer.count_parameters(pruned_model.model)
        compression_ratio = original_params / pruned_params if pruned_params > 0 else float('inf')
        sparsity = ModelAnalyzer.calculate_sparsity(pruned_model.model)
        
        return {
            'accuracy': {
                'original': original_results,
                'pruned': pruned_results,
                'accuracy_retention': {
                    key: 1.0 - abs(pruned_results['stereo_metrics'][key] - original_results['stereo_metrics'][key]) / original_results['stereo_metrics'][key]
                    for key in original_results['stereo_metrics']
                    if key in pruned_results['stereo_metrics'] and original_results['stereo_metrics'][key] != 0
                }
            },
            'performance': {
                'original_timing': original_timing,
                'pruned_timing': pruned_timing,
                'speedup': original_timing['mean_time'] / pruned_timing['mean_time'],
                'fps_improvement': pruned_timing['fps'] / original_timing['fps']
            },
            'memory': {
                'original_memory': original_memory,
                'pruned_memory': pruned_memory,
                'memory_reduction': 1.0 - pruned_memory['memory_allocated'] / original_memory['memory_allocated']
                if original_memory['memory_allocated'] > 0 else 0.0
            },
            'model_stats': {
                'original_params': original_params,
                'pruned_params': pruned_params,
                'compression_ratio': compression_ratio,
                'sparsity': sparsity,
                'size_reduction': 1.0 - pruned_params / original_params
            }
        }
    
    def generate_evaluation_report(self, comparison_results: Dict) -> str:
        """Generate a detailed evaluation report."""
        report = []
        report.append("="*60)
        report.append("PRUNED LIGHTSTEREO EVALUATION REPORT")
        report.append("="*60)
        
        # Model Statistics
        stats = comparison_results['model_stats']
        report.append("\nMODEL COMPRESSION:")
        report.append(f"  Original parameters: {stats['original_params']:,}")
        report.append(f"  Pruned parameters:   {stats['pruned_params']:,}")
        report.append(f"  Compression ratio:   {stats['compression_ratio']:.2f}x")
        report.append(f"  Sparsity level:      {stats['sparsity']:.1%}")
        report.append(f"  Size reduction:      {stats['size_reduction']:.1%}")
        
        # Performance
        perf = comparison_results['performance']
        report.append("\nPERFORMANCE:")
        report.append(f"  Original inference:  {perf['original_timing']['mean_time']*1000:.1f} ms")
        report.append(f"  Pruned inference:    {perf['pruned_timing']['mean_time']*1000:.1f} ms")
        report.append(f"  Speedup:             {perf['speedup']:.2f}x")
        report.append(f"  Original FPS:        {perf['original_timing']['fps']:.1f}")
        report.append(f"  Pruned FPS:          {perf['pruned_timing']['fps']:.1f}")
        
        # Memory
        memory = comparison_results['memory']
        report.append("\nMEMORY USAGE:")
        report.append(f"  Original memory:     {memory['original_memory']['memory_allocated']:.1f} MB")
        report.append(f"  Pruned memory:       {memory['pruned_memory']['memory_allocated']:.1f} MB")
        report.append(f"  Memory reduction:    {memory['memory_reduction']:.1%}")
        
        # Accuracy
        acc = comparison_results['accuracy']
        report.append("\nACCURACY METRICS:")
        orig_metrics = acc['original']['stereo_metrics']
        pruned_metrics = acc['pruned']['stereo_metrics']
        
        for metric in ['mae', 'rmse', 'bad_1', 'bad_3']:
            if metric in orig_metrics and metric in pruned_metrics:
                orig_val = orig_metrics[metric]
                pruned_val = pruned_metrics[metric]
                change = ((pruned_val - orig_val) / orig_val * 100) if orig_val != 0 else 0
                report.append(f"  {metric.upper():8s}: {orig_val:.3f} → {pruned_val:.3f} ({change:+.1f}%)")
        
        report.append("\n" + "="*60)
        
        return "\n".join(report)
    
    def plot_comparison(self, comparison_results: Dict, save_path: str = None):
        """Create comparison plots."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Performance comparison
        models = ['Original', 'Pruned']
        inference_times = [
            comparison_results['performance']['original_timing']['mean_time'] * 1000,
            comparison_results['performance']['pruned_timing']['mean_time'] * 1000
        ]
        
        ax1.bar(models, inference_times, color=['blue', 'red'])
        ax1.set_ylabel('Inference Time (ms)')
        ax1.set_title('Inference Speed Comparison')
        
        # Memory usage
        memory_usage = [
            comparison_results['memory']['original_memory']['memory_allocated'],
            comparison_results['memory']['pruned_memory']['memory_allocated']
        ]
        
        ax2.bar(models, memory_usage, color=['blue', 'red'])
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_title('Memory Usage Comparison')
        
        # Model size
        model_sizes = [
            comparison_results['model_stats']['original_params'] / 1e6,
            comparison_results['model_stats']['pruned_params'] / 1e6
        ]
        
        ax3.bar(models, model_sizes, color=['blue', 'red'])
        ax3.set_ylabel('Parameters (Millions)')
        ax3.set_title('Model Size Comparison')
        
        # Accuracy metrics
        metrics = ['mae', 'rmse', 'bad_1', 'bad_3']
        orig_acc = comparison_results['accuracy']['original']['stereo_metrics']
        pruned_acc = comparison_results['accuracy']['pruned']['stereo_metrics']
        
        x = np.arange(len(metrics))
        width = 0.35
        
        orig_vals = [orig_acc.get(m, 0) for m in metrics]
        pruned_vals = [pruned_acc.get(m, 0) for m in metrics]
        
        ax4.bar(x - width/2, orig_vals, width, label='Original', color='blue')
        ax4.bar(x + width/2, pruned_vals, width, label='Pruned', color='red')
        ax4.set_xlabel('Metrics')
        ax4.set_ylabel('Value')
        ax4.set_title('Accuracy Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(metrics)
        ax4.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def evaluate_pruned_lightstereo(
    original_model: LightStereo,
    pruned_model: AdvancedPrunedLightStereo,
    dataloader: DataLoader,
    device: torch.device,
    generate_report: bool = True,
    save_plots: str = None
) -> Dict:
    """
    Convenience function for comprehensive evaluation.
    
    Args:
        original_model: Original LightStereo model
        pruned_model: Pruned LightStereo model
        dataloader: Evaluation data loader
        device: Device to run on
        generate_report: Whether to generate text report
        save_plots: Path to save comparison plots
    
    Returns:
        Evaluation results
    """
    evaluator = PrunedModelEvaluator(device)
    results = evaluator.compare_models(original_model, pruned_model, dataloader)
    
    if generate_report:
        report = evaluator.generate_evaluation_report(results)
        print(report)
    
    if save_plots:
        evaluator.plot_comparison(results, save_plots)
    
    return results