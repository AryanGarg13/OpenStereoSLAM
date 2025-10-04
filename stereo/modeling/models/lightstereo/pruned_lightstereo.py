"""
Pruned LightStereo model using magnitude-based weight pruning.
Wraps the original LightStereo model and applies weight masks during forward pass.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from .lightstereo import LightStereo
from .weight_pruning import create_weight_masks, apply_weight_masks, get_sparsity_stats


class PrunedLightStereo(nn.Module):
    """
    Weight-pruned version of LightStereo that maintains original architecture
    but zeros out weights below a magnitude threshold.
    """
    
    def __init__(self, base_model: LightStereo, pruning_config: Dict):
        """
        Initialize pruned model.
        
        Args:
            base_model: Original trained LightStereo model
            pruning_config: Configuration dictionary with pruning parameters
        """
        super().__init__()
        
        # Store the base model
        self.base_model = base_model
        
        # Pruning configuration
        self.pruning_ratio = pruning_config.get('pruning_ratio', 0.5)
        self.pruning_type = pruning_config.get('pruning_type', 'global')  # 'global' or 'layerwise'
        self.structured = pruning_config.get('structured', False)
        
        # Create and store weight masks
        self.weight_masks = self._create_weight_masks()
        
        # Apply masks to base model
        self._apply_masks()
        
        # Store sparsity statistics
        self.sparsity_stats = get_sparsity_stats(self.base_model, self.weight_masks)
    
    def _create_weight_masks(self) -> Dict[str, torch.Tensor]:
        """Create weight masks based on pruning configuration."""
        if self.structured:
            from .weight_pruning import structured_channel_pruning
            return structured_channel_pruning(self.base_model, self.pruning_ratio)
        else:
            return create_weight_masks(self.base_model, self.pruning_ratio, self.pruning_type)
    
    def _apply_masks(self) -> None:
        """Apply weight masks to the base model."""
        apply_weight_masks(self.base_model, self.weight_masks)
    
    def forward(self, data):
        """Forward pass through the pruned model."""
        # Ensure masks are applied before each forward pass
        self._apply_masks()
        return self.base_model(data)
    
    def get_loss(self, model_pred, input_data):
        """Get loss using the base model's loss function."""
        return self.base_model.get_loss(model_pred, input_data)
    
    def get_sparsity_info(self) -> Dict:
        """Get detailed sparsity information."""
        return self.sparsity_stats
    
    def update_pruning(self, new_pruning_ratio: float, new_pruning_type: str = None):
        """
        Update pruning with new ratio and/or type.
        
        Args:
            new_pruning_ratio: New pruning ratio
            new_pruning_type: New pruning type ('global' or 'layerwise')
        """
        self.pruning_ratio = new_pruning_ratio
        if new_pruning_type is not None:
            self.pruning_type = new_pruning_type
        
        # Recreate masks with new settings
        self.weight_masks = self._create_weight_masks()
        self._apply_masks()
        self.sparsity_stats = get_sparsity_stats(self.base_model, self.weight_masks)
    
    @property
    def max_disp(self):
        """Delegate to base model."""
        return self.base_model.max_disp
    
    @property
    def left_att(self):
        """Delegate to base model."""
        return self.base_model.left_att
    
    def train(self, mode: bool = True):
        """Set training mode."""
        super().train(mode)
        self.base_model.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode."""
        super().eval()
        self.base_model.eval()
        return self
    
    def to(self, device):
        """Move model to device."""
        super().to(device)
        self.base_model.to(device)
        # Move masks to device
        for name in self.weight_masks:
            self.weight_masks[name] = self.weight_masks[name].to(device)
        return self
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Get state dict including masks."""
        state_dict = self.base_model.state_dict(destination, prefix, keep_vars)
        # Add masks to state dict
        for name, mask in self.weight_masks.items():
            state_dict[f'{prefix}masks.{name}'] = mask
        return state_dict
    
    def load_state_dict(self, state_dict, strict: bool = True):
        """Load state dict including masks."""
        # Separate base model state dict and masks
        base_state_dict = {}
        mask_state_dict = {}
        
        for key, value in state_dict.items():
            if key.startswith('masks.'):
                mask_key = key[6:]  # Remove 'masks.' prefix
                mask_state_dict[mask_key] = value
            else:
                base_state_dict[key] = value
        
        # Load base model state
        self.base_model.load_state_dict(base_state_dict, strict=strict)
        
        # Load masks if available
        if mask_state_dict:
            self.weight_masks.update(mask_state_dict)
            self._apply_masks()
            self.sparsity_stats = get_sparsity_stats(self.base_model, self.weight_masks)


def create_pruned_model(base_model: LightStereo, 
                       pruning_ratio: float = 0.5,
                       pruning_type: str = 'global',
                       structured: bool = False) -> PrunedLightStereo:
    """
    Factory function to create a pruned LightStereo model.
    
    Args:
        base_model: Original trained LightStereo model
        pruning_ratio: Fraction of weights to prune (0.0 to 1.0)
        pruning_type: 'global' or 'layerwise'
        structured: Whether to use structured pruning
        
    Returns:
        Pruned LightStereo model
    """
    pruning_config = {
        'pruning_ratio': pruning_ratio,
        'pruning_type': pruning_type,
        'structured': structured
    }
    
    return PrunedLightStereo(base_model, pruning_config)


def create_multiple_pruned_models(base_model: LightStereo, 
                                 pruning_ratios: list = [0.3, 0.5, 0.7]) -> Dict[str, PrunedLightStereo]:
    """
    Create multiple pruned models with different pruning ratios.
    
    Args:
        base_model: Original trained LightStereo model
        pruning_ratios: List of pruning ratios to create
        
    Returns:
        Dictionary mapping pruning level names to pruned models
    """
    level_names = {0.3: 'light', 0.5: 'medium', 0.7: 'aggressive'}
    pruned_models = {}
    
    for ratio in pruning_ratios:
        level_name = level_names.get(ratio, f'custom_{ratio}')
        pruned_models[level_name] = create_pruned_model(base_model, ratio, 'global', False)
    
    return pruned_models


class GradualPruner:
    """
    Utility class for gradual pruning during training/fine-tuning.
    """
    
    def __init__(self, model: LightStereo, 
                 initial_ratio: float = 0.0,
                 final_ratio: float = 0.5,
                 num_steps: int = 10):
        """
        Initialize gradual pruner.
        
        Args:
            model: LightStereo model to prune
            initial_ratio: Starting pruning ratio
            final_ratio: Final pruning ratio
            num_steps: Number of pruning steps
        """
        self.model = model
        self.initial_ratio = initial_ratio
        self.final_ratio = final_ratio
        self.num_steps = num_steps
        
        from .weight_pruning import gradual_pruning_schedule
        self.schedule = gradual_pruning_schedule(initial_ratio, final_ratio, num_steps)
        self.current_step = 0
        
        # Create initial masks
        self.current_masks = create_weight_masks(model, initial_ratio, 'global')
    
    def step(self):
        """Perform one pruning step."""
        if self.current_step < len(self.schedule):
            ratio = self.schedule[self.current_step]
            self.current_masks = create_weight_masks(self.model, ratio, 'global')
            apply_weight_masks(self.model, self.current_masks)
            self.current_step += 1
    
    def get_current_sparsity(self) -> float:
        """Get current model sparsity."""
        stats = get_sparsity_stats(self.model, self.current_masks)
        return stats['overall_sparsity']
    
    def is_complete(self) -> bool:
        """Check if pruning schedule is complete."""
        return self.current_step >= len(self.schedule)