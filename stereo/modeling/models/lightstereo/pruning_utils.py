import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


class PruningUtils:
    @staticmethod
    def magnitude_based_pruning(model: nn.Module, sparsity: float, exclude_layers: List[str] = None) -> Dict[str, torch.Tensor]:
        """
        Apply magnitude-based unstructured pruning to model weights.
        
        Args:
            model: PyTorch model to prune
            sparsity: Fraction of weights to prune (0.0 to 1.0)
            exclude_layers: List of layer names to exclude from pruning
            
        Returns:
            Dictionary of pruning masks for each layer
        """
        if exclude_layers is None:
            exclude_layers = []
            
        masks = {}
        all_weights = []
        
        # Collect all weights for global magnitude threshold
        for name, param in model.named_parameters():
            if 'weight' in name and not any(exclude in name for exclude in exclude_layers):
                all_weights.append(param.data.abs().flatten())
        
        if not all_weights:
            return masks
            
        # Calculate global threshold
        all_weights_tensor = torch.cat(all_weights)
        threshold = torch.quantile(all_weights_tensor, sparsity)
        
        # Create masks
        for name, param in model.named_parameters():
            if 'weight' in name and not any(exclude in name for exclude in exclude_layers):
                mask = (param.data.abs() > threshold).float()
                masks[name] = mask
                
        return masks
    
    @staticmethod
    def apply_pruning_masks(model: nn.Module, masks: Dict[str, torch.Tensor]):
        """Apply pruning masks to model parameters."""
        for name, param in model.named_parameters():
            if name in masks:
                param.data *= masks[name]
    
    @staticmethod
    def structured_channel_pruning(model: nn.Module, layer_name: str, channels_to_remove: List[int]) -> nn.Module:
        """
        Remove specific channels from a convolutional layer.
        
        Args:
            model: PyTorch model
            layer_name: Name of layer to prune
            channels_to_remove: List of channel indices to remove
            
        Returns:
            Model with pruned channels
        """
        def get_layer_by_name(model, name):
            parts = name.split('.')
            layer = model
            for part in parts:
                if part.isdigit():
                    layer = layer[int(part)]
                else:
                    layer = getattr(layer, part)
            return layer
        
        def set_layer_by_name(model, name, new_layer):
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                if part.isdigit():
                    parent = parent[int(part)]
                else:
                    parent = getattr(parent, part)
            
            if parts[-1].isdigit():
                parent[int(parts[-1])] = new_layer
            else:
                setattr(parent, parts[-1], new_layer)
        
        layer = get_layer_by_name(model, layer_name)
        
        if isinstance(layer, nn.Conv2d):
            # Create new conv layer with reduced channels
            remaining_channels = [i for i in range(layer.out_channels) if i not in channels_to_remove]
            
            new_conv = nn.Conv2d(
                layer.in_channels,
                len(remaining_channels),
                layer.kernel_size,
                layer.stride,
                layer.padding,
                layer.dilation,
                layer.groups,
                bias=layer.bias is not None
            )
            
            # Copy weights for remaining channels
            new_conv.weight.data = layer.weight.data[remaining_channels]
            if layer.bias is not None:
                new_conv.bias.data = layer.bias.data[remaining_channels]
                
            set_layer_by_name(model, layer_name, new_conv)
            
        return model
    
    @staticmethod
    def calculate_channel_importance(layer: nn.Module, importance_metric: str = 'l1') -> torch.Tensor:
        """
        Calculate importance scores for channels in a layer.
        
        Args:
            layer: Convolutional layer
            importance_metric: 'l1', 'l2', or 'geometric_median'
            
        Returns:
            Importance scores for each output channel
        """
        if not isinstance(layer, (nn.Conv2d, nn.Linear)):
            raise ValueError("Layer must be Conv2d or Linear")
            
        weights = layer.weight.data
        
        if importance_metric == 'l1':
            # L1 norm of filters
            importance = weights.abs().sum(dim=(1, 2, 3) if len(weights.shape) == 4 else 1)
        elif importance_metric == 'l2':
            # L2 norm of filters
            importance = weights.pow(2).sum(dim=(1, 2, 3) if len(weights.shape) == 4 else 1).sqrt()
        elif importance_metric == 'geometric_median':
            # Geometric median distance
            importance = torch.zeros(weights.shape[0], device=weights.device)
            weights_flat = weights.view(weights.shape[0], -1)
            
            for i in range(weights.shape[0]):
                distances = torch.norm(weights_flat - weights_flat[i], dim=1)
                importance[i] = distances.sum()
        else:
            raise ValueError(f"Unknown importance metric: {importance_metric}")
            
        return importance
    
    @staticmethod
    def get_channels_to_prune(layer: nn.Module, prune_ratio: float, importance_metric: str = 'l1') -> List[int]:
        """
        Get list of channel indices to prune based on importance scores.
        
        Args:
            layer: Convolutional layer
            prune_ratio: Fraction of channels to prune
            importance_metric: Method to calculate importance
            
        Returns:
            List of channel indices to remove
        """
        importance = PruningUtils.calculate_channel_importance(layer, importance_metric)
        num_channels_to_prune = int(layer.out_channels * prune_ratio)
        
        # Get indices of least important channels
        _, indices = importance.sort()
        channels_to_prune = indices[:num_channels_to_prune].tolist()
        
        return channels_to_prune


class PruningScheduler:
    """Manages progressive pruning schedule during training."""
    
    def __init__(self, initial_sparsity: float = 0.0, final_sparsity: float = 0.5, 
                 pruning_frequency: int = 100, pruning_steps: int = 10):
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.pruning_frequency = pruning_frequency
        self.pruning_steps = pruning_steps
        self.current_step = 0
        self.pruning_step = 0
        
    def should_prune(self, step: int) -> bool:
        """Check if pruning should be applied at current step."""
        return step % self.pruning_frequency == 0 and self.pruning_step < self.pruning_steps
    
    def get_current_sparsity(self) -> float:
        """Get current sparsity level based on schedule."""
        if self.pruning_step >= self.pruning_steps:
            return self.final_sparsity
            
        progress = self.pruning_step / self.pruning_steps
        # Polynomial sparsity schedule (cubic)
        sparsity = self.initial_sparsity + (self.final_sparsity - self.initial_sparsity) * (3 * progress**2 - 2 * progress**3)
        return sparsity
    
    def step(self):
        """Increment pruning step counter."""
        self.pruning_step += 1


class ModelAnalyzer:
    """Analyze model structure and pruning effects."""
    
    @staticmethod
    def count_parameters(model: nn.Module, only_trainable: bool = True) -> int:
        """Count total parameters in model."""
        if only_trainable:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in model.parameters())
    
    @staticmethod
    def calculate_sparsity(model: nn.Module) -> float:
        """Calculate current sparsity level of model."""
        total_params = 0
        zero_params = 0
        
        for param in model.parameters():
            if param.requires_grad and 'weight' in str(param):
                total_params += param.numel()
                zero_params += (param.abs() < 1e-8).sum().item()
        
        return zero_params / total_params if total_params > 0 else 0.0
    
    @staticmethod
    def analyze_layer_sizes(model: nn.Module) -> Dict[str, Tuple[int, int]]:
        """Analyze input/output sizes for each layer."""
        layer_info = {}
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                layer_info[name] = (module.in_channels, module.out_channels)
            elif isinstance(module, nn.Linear):
                layer_info[name] = (module.in_features, module.out_features)
                
        return layer_info
    
    @staticmethod
    def estimate_flops_reduction(original_model: nn.Module, pruned_model: nn.Module, 
                                input_shape: Tuple[int, ...]) -> float:
        """Estimate FLOPS reduction from pruning."""
        # This is a simplified estimation
        orig_params = ModelAnalyzer.count_parameters(original_model)
        pruned_params = ModelAnalyzer.count_parameters(pruned_model)
        
        reduction_ratio = 1.0 - (pruned_params / orig_params)
        return reduction_ratio