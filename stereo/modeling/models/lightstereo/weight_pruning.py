"""
Weight-based pruning utilities for LightStereo models.
Implements magnitude-based post-training pruning.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union


def get_model_weights(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extract all weights from model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary mapping parameter names to weight tensors
    """
    weights = {}
    for name, param in model.named_parameters():
        if param.requires_grad and len(param.shape) > 1:  # Only weight matrices, not biases
            weights[name] = param.data.clone()
    return weights


def compute_global_threshold(model: nn.Module, pruning_ratio: float) -> float:
    """
    Compute global magnitude threshold for pruning.
    
    Args:
        model: PyTorch model
        pruning_ratio: Fraction of weights to prune (0.0 to 1.0)
        
    Returns:
        Threshold value - weights below this will be pruned
    """
    all_weights = []
    
    for name, param in model.named_parameters():
        if param.requires_grad and len(param.shape) > 1:
            all_weights.append(param.data.abs().flatten())
    
    if not all_weights:
        return 0.0
    
    all_weights = torch.cat(all_weights)
    k = int(len(all_weights) * pruning_ratio)
    
    if k == 0:
        return 0.0
    
    threshold, _ = torch.kthvalue(all_weights, k)
    return threshold.item()


def compute_layerwise_thresholds(model: nn.Module, pruning_ratio: float) -> Dict[str, float]:
    """
    Compute layer-wise magnitude thresholds for pruning.
    
    Args:
        model: PyTorch model
        pruning_ratio: Fraction of weights to prune per layer
        
    Returns:
        Dictionary mapping parameter names to threshold values
    """
    thresholds = {}
    
    for name, param in model.named_parameters():
        if param.requires_grad and len(param.shape) > 1:
            weights = param.data.abs().flatten()
            k = int(len(weights) * pruning_ratio)
            
            if k == 0:
                thresholds[name] = 0.0
            else:
                threshold, _ = torch.kthvalue(weights, k)
                thresholds[name] = threshold.item()
    
    return thresholds


def create_weight_masks(model: nn.Module, 
                       pruning_ratio: float,
                       pruning_type: str = "global") -> Dict[str, torch.Tensor]:
    """
    Create binary masks for weight pruning.
    
    Args:
        model: PyTorch model
        pruning_ratio: Fraction of weights to prune
        pruning_type: "global" or "layerwise"
        
    Returns:
        Dictionary mapping parameter names to binary masks
    """
    masks = {}
    
    if pruning_type == "global":
        threshold = compute_global_threshold(model, pruning_ratio)
        for name, param in model.named_parameters():
            if param.requires_grad and len(param.shape) > 1:
                masks[name] = (param.data.abs() >= threshold).float()
    
    elif pruning_type == "layerwise":
        thresholds = compute_layerwise_thresholds(model, pruning_ratio)
        for name, param in model.named_parameters():
            if param.requires_grad and len(param.shape) > 1:
                threshold = thresholds[name]
                masks[name] = (param.data.abs() >= threshold).float()
    
    else:
        raise ValueError(f"Unknown pruning type: {pruning_type}")
    
    return masks


def apply_weight_masks(model: nn.Module, masks: Dict[str, torch.Tensor]) -> None:
    """
    Apply weight masks to model parameters in-place.
    
    Args:
        model: PyTorch model
        masks: Dictionary of binary masks
    """
    for name, param in model.named_parameters():
        if name in masks:
            param.data *= masks[name]


def get_sparsity_stats(model: nn.Module, masks: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, float]:
    """
    Compute sparsity statistics for a model.
    
    Args:
        model: PyTorch model
        masks: Optional masks to compute effective sparsity
        
    Returns:
        Dictionary with sparsity statistics
    """
    total_params = 0
    zero_params = 0
    layer_stats = {}
    
    for name, param in model.named_parameters():
        if param.requires_grad and len(param.shape) > 1:
            if masks and name in masks:
                effective_weights = param.data * masks[name]
            else:
                effective_weights = param.data
            
            layer_total = effective_weights.numel()
            layer_zeros = (effective_weights == 0).sum().item()
            layer_sparsity = layer_zeros / layer_total
            
            layer_stats[name] = {
                'total_params': layer_total,
                'zero_params': layer_zeros,
                'sparsity': layer_sparsity
            }
            
            total_params += layer_total
            zero_params += layer_zeros
    
    overall_sparsity = zero_params / total_params if total_params > 0 else 0.0
    
    return {
        'overall_sparsity': overall_sparsity,
        'total_params': total_params,
        'zero_params': zero_params,
        'layer_stats': layer_stats
    }


def structured_channel_pruning(model: nn.Module, 
                              pruning_ratio: float,
                              layer_types: List[type] = None) -> Dict[str, torch.Tensor]:
    """
    Create masks for structured channel pruning.
    
    Args:
        model: PyTorch model
        pruning_ratio: Fraction of channels to prune
        layer_types: List of layer types to consider for pruning
        
    Returns:
        Dictionary of channel masks
    """
    if layer_types is None:
        layer_types = [nn.Conv2d, nn.Linear]
    
    masks = {}
    
    for name, module in model.named_modules():
        if any(isinstance(module, layer_type) for layer_type in layer_types):
            if hasattr(module, 'weight') and module.weight is not None:
                weight = module.weight.data
                
                # Compute channel importance (L2 norm)
                if len(weight.shape) == 4:  # Conv2D
                    channel_norms = torch.norm(weight, dim=(1, 2, 3))
                elif len(weight.shape) == 2:  # Linear
                    channel_norms = torch.norm(weight, dim=1)
                else:
                    continue
                
                num_channels = len(channel_norms)
                num_prune = int(num_channels * pruning_ratio)
                
                if num_prune > 0:
                    _, indices_to_prune = torch.topk(channel_norms, num_prune, largest=False)
                    
                    # Create mask
                    channel_mask = torch.ones(num_channels, device=weight.device)
                    channel_mask[indices_to_prune] = 0
                    
                    # Expand mask to weight dimensions
                    if len(weight.shape) == 4:
                        mask = channel_mask.view(-1, 1, 1, 1).expand_as(weight)
                    else:
                        mask = channel_mask.view(-1, 1).expand_as(weight)
                    
                    masks[f"{name}.weight"] = mask
    
    return masks


def gradual_pruning_schedule(initial_ratio: float, 
                           final_ratio: float, 
                           num_steps: int) -> List[float]:
    """
    Create a gradual pruning schedule.
    
    Args:
        initial_ratio: Starting pruning ratio
        final_ratio: Final pruning ratio
        num_steps: Number of pruning steps
        
    Returns:
        List of pruning ratios for each step
    """
    if num_steps == 1:
        return [final_ratio]
    
    ratios = []
    for i in range(num_steps):
        progress = i / (num_steps - 1)
        ratio = initial_ratio + progress * (final_ratio - initial_ratio)
        ratios.append(ratio)
    
    return ratios


class PruningAnalyzer:
    """Utility class for analyzing pruning effects."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.original_weights = get_model_weights(model)
    
    def analyze_pruning_candidates(self, pruning_ratio: float) -> Dict:
        """Analyze which weights would be pruned at given ratio."""
        threshold = compute_global_threshold(self.model, pruning_ratio)
        
        analysis = {
            'threshold': threshold,
            'layer_analysis': {}
        }
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and len(param.shape) > 1:
                weights = param.data.abs()
                would_prune = weights < threshold
                
                analysis['layer_analysis'][name] = {
                    'total_weights': weights.numel(),
                    'weights_to_prune': would_prune.sum().item(),
                    'pruning_ratio': would_prune.float().mean().item(),
                    'min_weight': weights.min().item(),
                    'max_weight': weights.max().item(),
                    'mean_weight': weights.mean().item()
                }
        
        return analysis
    
    def compare_pruning_methods(self, pruning_ratio: float) -> Dict:
        """Compare global vs layerwise pruning."""
        global_masks = create_weight_masks(self.model, pruning_ratio, "global")
        layerwise_masks = create_weight_masks(self.model, pruning_ratio, "layerwise")
        
        global_stats = get_sparsity_stats(self.model, global_masks)
        layerwise_stats = get_sparsity_stats(self.model, layerwise_masks)
        
        return {
            'global_pruning': global_stats,
            'layerwise_pruning': layerwise_stats
        }