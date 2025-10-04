"""
Hybrid Pruned LightStereo model with 60% unstructured + 40% structured pruning.
Advanced pruning implementation with structured channel pruning and unstructured weight pruning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from typing import Dict, List, Tuple, Optional

from .lightstereo import LightStereo
from .pruning_utils import PruningUtils, PruningScheduler, ModelAnalyzer


class AdvancedPrunedLightStereo(nn.Module):
    """
    Advanced pruned version of LightStereo with hybrid structured + unstructured pruning.
    Supports 60% unstructured + 40% structured pruning strategy.
    """
    
    def __init__(self, original_model: LightStereo, pruning_config: Dict = None):
        super().__init__()
        
        # Default pruning configuration (60% unstructured, 40% structured)
        self.default_config = {
            'unstructured_ratios': {
                'backbone': 0.4,
                'cost_agg': 0.3,
                'refine_1': 0.5,
            },
            'structured_ratios': {
                'backbone_channels': 0.25,
                'cost_agg_channels': 0.15,
                'refine_2_channels': 0.3,
                'refine_3_channels': 0.4,
            },
            'importance_metric': 'l1',
            'exclude_layers': ['stem_2']  # Keep input processing intact
        }
        
        self.pruning_config = pruning_config if pruning_config else self.default_config
        self.original_model = original_model
        self.max_disp = original_model.max_disp
        self.left_att = original_model.left_att
        self.training = original_model.training
        
        # Create pruned model
        self.model = self._create_pruned_model()
        self.pruning_masks = {}
        self.is_pruned = False
        
    def _create_pruned_model(self) -> LightStereo:
        """Create a copy of the original model for pruning."""
        return deepcopy(self.original_model)
    
    def apply_structured_pruning(self):
        """Apply structured pruning to remove channels/filters."""
        print("Applying structured pruning...")
        
        structured_ratios = self.pruning_config['structured_ratios']
        importance_metric = self.pruning_config['importance_metric']
        
        # Prune backbone output channels
        if 'backbone_channels' in structured_ratios:
            self._prune_backbone_channels(structured_ratios['backbone_channels'], importance_metric)
        
        # Prune cost aggregation channels
        if 'cost_agg_channels' in structured_ratios:
            self._prune_cost_agg_channels(structured_ratios['cost_agg_channels'], importance_metric)
        
        # Prune refinement channels
        if 'refine_2_channels' in structured_ratios:
            self._prune_refine_channels(structured_ratios['refine_2_channels'], importance_metric)
        
        # Prune final deconv channels
        if 'refine_3_channels' in structured_ratios:
            self._prune_final_channels(structured_ratios['refine_3_channels'], importance_metric)
    
    def _prune_backbone_channels(self, prune_ratio: float, importance_metric: str):
        """Prune channels from backbone output."""
        # Get the last layer of backbone that affects cost aggregation
        backbone_layers = []
        for name, module in self.model.backbone.named_modules():
            if isinstance(module, nn.Conv2d):
                backbone_layers.append((name, module))
        
        if backbone_layers:
            last_layer_name, last_layer = backbone_layers[-1]
            channels_to_prune = PruningUtils.get_channels_to_prune(
                last_layer, prune_ratio, importance_metric
            )
            
            # Update backbone output channels
            remaining_channels = [i for i in range(last_layer.out_channels) if i not in channels_to_prune]
            self.model.backbone.output_channels[0] = len(remaining_channels)
            
            # Prune the layer
            PruningUtils.structured_channel_pruning(
                self.model.backbone, last_layer_name, channels_to_prune
            )
            
            # Update dependent layers (cost_agg and refine_1)
            self._update_dependent_layers_after_backbone_pruning(len(remaining_channels))
    
    def _prune_cost_agg_channels(self, prune_ratio: float, importance_metric: str):
        """Prune channels from cost aggregation module."""
        # Find first conv layer in cost aggregation
        for name, module in self.model.cost_agg.named_modules():
            if isinstance(module, nn.Conv2d):
                channels_to_prune = PruningUtils.get_channels_to_prune(
                    module, prune_ratio, importance_metric
                )
                PruningUtils.structured_channel_pruning(
                    self.model.cost_agg, name, channels_to_prune
                )
                break
    
    def _prune_refine_channels(self, prune_ratio: float, importance_metric: str):
        """Prune channels from refinement layers."""
        # Prune refine_2 (FPNLayer)
        if hasattr(self.model.refine_2, 'conv'):
            channels_to_prune = PruningUtils.get_channels_to_prune(
                self.model.refine_2.conv, prune_ratio, importance_metric
            )
            PruningUtils.structured_channel_pruning(
                self.model, 'refine_2.conv', channels_to_prune
            )
    
    def _prune_final_channels(self, prune_ratio: float, importance_metric: str):
        """Prune channels from final deconv layer."""
        # Prune refine_3 (BasicDeconv2d)
        if hasattr(self.model.refine_3, 'deconv'):
            channels_to_prune = PruningUtils.get_channels_to_prune(
                self.model.refine_3.deconv, prune_ratio, importance_metric
            )
            PruningUtils.structured_channel_pruning(
                self.model, 'refine_3.deconv', channels_to_prune
            )
    
    def _update_dependent_layers_after_backbone_pruning(self, new_channel_count: int):
        """Update layers that depend on backbone output channels."""
        # Update cost aggregation input
        if hasattr(self.model.cost_agg, 'backbone_channels'):
            self.model.cost_agg.backbone_channels = new_channel_count
        
        # Update refine_1 input channels
        if isinstance(self.model.refine_1[0], nn.Conv2d):
            old_conv = self.model.refine_1[0]
            new_conv = nn.Conv2d(
                new_channel_count,  # Updated input channels
                old_conv.out_channels,
                old_conv.kernel_size,
                old_conv.stride,
                old_conv.padding,
                bias=old_conv.bias is not None
            )
            
            # Copy weights (truncate if necessary)
            channels_to_copy = min(new_channel_count, old_conv.in_channels)
            new_conv.weight.data = old_conv.weight.data[:, :channels_to_copy]
            if old_conv.bias is not None:
                new_conv.bias.data = old_conv.bias.data
                
            self.model.refine_1[0] = new_conv
    
    def apply_unstructured_pruning(self, global_sparsity: float = None):
        """Apply magnitude-based unstructured pruning."""
        print("Applying unstructured pruning...")
        
        unstructured_ratios = self.pruning_config['unstructured_ratios']
        exclude_layers = self.pruning_config.get('exclude_layers', [])
        
        if global_sparsity:
            # Apply global sparsity
            self.pruning_masks = PruningUtils.magnitude_based_pruning(
                self.model, global_sparsity, exclude_layers
            )
        else:
            # Apply layer-specific sparsity
            self.pruning_masks = {}
            
            # Backbone pruning
            if 'backbone' in unstructured_ratios:
                backbone_masks = PruningUtils.magnitude_based_pruning(
                    self.model.backbone, unstructured_ratios['backbone'], exclude_layers
                )
                self.pruning_masks.update(backbone_masks)
            
            # Cost aggregation pruning
            if 'cost_agg' in unstructured_ratios:
                cost_agg_masks = PruningUtils.magnitude_based_pruning(
                    self.model.cost_agg, unstructured_ratios['cost_agg'], exclude_layers
                )
                self.pruning_masks.update(cost_agg_masks)
            
            # Refinement pruning
            if 'refine_1' in unstructured_ratios:
                refine_masks = PruningUtils.magnitude_based_pruning(
                    self.model.refine_1, unstructured_ratios['refine_1'], exclude_layers
                )
                self.pruning_masks.update(refine_masks)
        
        # Apply masks
        PruningUtils.apply_pruning_masks(self.model, self.pruning_masks)
    
    def apply_full_pruning(self, global_unstructured_sparsity: float = None):
        """Apply both structured and unstructured pruning."""
        print("Starting hybrid pruning (60% unstructured + 40% structured)...")
        
        # Step 1: Apply structured pruning first
        self.apply_structured_pruning()
        
        # Step 2: Apply unstructured pruning
        self.apply_unstructured_pruning(global_unstructured_sparsity)
        
        self.is_pruned = True
        
        # Print pruning statistics
        self._print_pruning_stats()
    
    def _print_pruning_stats(self):
        """Print pruning statistics."""
        original_params = ModelAnalyzer.count_parameters(self.original_model)
        pruned_params = ModelAnalyzer.count_parameters(self.model)
        sparsity = ModelAnalyzer.calculate_sparsity(self.model)
        
        compression_ratio = original_params / pruned_params if pruned_params > 0 else float('inf')
        
        print(f"\n=== Pruning Statistics ===")
        print(f"Original parameters: {original_params:,}")
        print(f"Pruned parameters: {pruned_params:,}")
        print(f"Compression ratio: {compression_ratio:.2f}x")
        print(f"Sparsity level: {sparsity:.1%}")
        print(f"Parameter reduction: {(1 - pruned_params/original_params):.1%}")
    
    def forward(self, data):
        """Forward pass with pruning masks applied."""
        if self.is_pruned and self.pruning_masks:
            # Ensure masks are applied before forward pass
            PruningUtils.apply_pruning_masks(self.model, self.pruning_masks)
        
        return self.model(data)
    
    def get_loss(self, model_pred, input_data):
        """Compute loss using original model's loss function."""
        return self.model.get_loss(model_pred, input_data)
    
    def remove_pruning(self):
        """Remove pruning masks and restore original weights."""
        if hasattr(self, 'original_weights'):
            for name, param in self.model.named_parameters():
                if name in self.original_weights:
                    param.data = self.original_weights[name].clone()
        
        self.pruning_masks = {}
        self.is_pruned = False
    
    def make_pruning_permanent(self):
        """Make pruning permanent by removing pruned weights completely."""
        if not self.is_pruned:
            return
        
        for name, param in self.model.named_parameters():
            if name in self.pruning_masks:
                # Zero out pruned weights permanently
                param.data *= self.pruning_masks[name]
        
        # Clear masks since pruning is now permanent
        self.pruning_masks = {}
    
    def get_model_size(self) -> Tuple[int, float]:
        """Get model size in parameters and MB."""
        params = ModelAnalyzer.count_parameters(self.model)
        size_mb = params * 4 / (1024 ** 2)  # Assuming float32
        return params, size_mb
    
    def save_pruned_model(self, filepath: str):
        """Save the pruned model state."""
        state = {
            'model_state_dict': self.model.state_dict(),
            'pruning_config': self.pruning_config,
            'pruning_masks': self.pruning_masks,
            'is_pruned': self.is_pruned,
            'max_disp': self.max_disp,
            'left_att': self.left_att
        }
        torch.save(state, filepath)
        print(f"Pruned model saved to {filepath}")
    
    @classmethod
    def load_pruned_model(cls, filepath: str, original_model: LightStereo):
        """Load a pruned model from file."""
        state = torch.load(filepath)
        
        pruned_model = cls(original_model, state['pruning_config'])
        pruned_model.model.load_state_dict(state['model_state_dict'])
        pruned_model.pruning_masks = state['pruning_masks']
        pruned_model.is_pruned = state['is_pruned']
        
        return pruned_model


class HybridPruningTrainer:
    """
    Training wrapper for progressive hybrid pruning with fine-tuning.
    """
    
    def __init__(self, model: LightStereo, pruning_config: Dict = None):
        self.original_model = model
        self.pruning_config = pruning_config
        self.pruned_model = None
        
        # Progressive pruning scheduler
        self.scheduler = PruningScheduler(
            initial_sparsity=0.0,
            final_sparsity=0.6,  # Final unstructured sparsity
            pruning_frequency=500,  # Steps between pruning
            pruning_steps=10
        )
    
    def create_pruned_model(self) -> AdvancedPrunedLightStereo:
        """Create the pruned model instance."""
        self.pruned_model = AdvancedPrunedLightStereo(
            self.original_model, 
            self.pruning_config
        )
        return self.pruned_model
    
    def progressive_pruning_step(self, step: int) -> bool:
        """
        Perform progressive pruning step during training.
        
        Returns:
            True if pruning was applied, False otherwise
        """
        if not self.pruned_model:
            return False
            
        if self.scheduler.should_prune(step):
            current_sparsity = self.scheduler.get_current_sparsity()
            
            print(f"Step {step}: Applying progressive pruning (sparsity: {current_sparsity:.1%})")
            
            # Apply current level of unstructured pruning
            self.pruned_model.apply_unstructured_pruning(current_sparsity)
            
            self.scheduler.step()
            return True
        
        return False
    
    def apply_full_hybrid_pruning(self):
        """Apply complete hybrid pruning strategy."""
        if self.pruned_model:
            self.pruned_model.apply_full_pruning()
        else:
            raise ValueError("Create pruned model first using create_pruned_model()")
    
    def get_training_statistics(self) -> Dict:
        """Get current training and pruning statistics."""
        if not self.pruned_model:
            return {}
        
        stats = {
            'current_sparsity': self.scheduler.get_current_sparsity(),
            'pruning_step': self.scheduler.pruning_step,
            'is_pruned': self.pruned_model.is_pruned,
            'model_size_mb': self.pruned_model.get_model_size()[1]
        }
        
        return stats


def create_hybrid_pruned_lightstereo(
    original_model: LightStereo,
    unstructured_ratios: Dict[str, float] = None,
    structured_ratios: Dict[str, float] = None
) -> AdvancedPrunedLightStereo:
    """
    Factory function to create hybrid pruned LightStereo model.
    
    Args:
        original_model: Original trained LightStereo model
        unstructured_ratios: Custom unstructured pruning ratios
        structured_ratios: Custom structured pruning ratios
    
    Returns:
        Hybrid pruned LightStereo model
    """
    config = {}
    
    if unstructured_ratios:
        config['unstructured_ratios'] = unstructured_ratios
    
    if structured_ratios:
        config['structured_ratios'] = structured_ratios
    
    return AdvancedPrunedLightStereo(original_model, config if config else None)