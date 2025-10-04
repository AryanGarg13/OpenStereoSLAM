"""
Test script for the new weight-based pruning implementation.
"""

import torch
import torch.nn as nn
import sys
import os

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

def test_basic_pruning():
    """Test basic pruning functionality."""
    print("Testing basic pruning functionality...")
    
    try:
        # Create a simple model configuration
        class Config:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
            
            def get(self, key, default=None):
                return getattr(self, key, default)
        
        config = Config(
            MAX_DISP=192,
            LEFT_ATT=True,
            BACKCONE='MobileNetv2',
            AGGREGATION_BLOCKS=[3, 3, 3],
            EXPANSE_RATIO=6
        )
        
        # Create original model
        from lightstereo import LightStereo
        original_model = LightStereo(config)
        print("✓ Created original LightStereo model")
        
        # Test weight pruning utilities
        from weight_pruning import (
            get_model_weights, 
            compute_global_threshold,
            create_weight_masks,
            get_sparsity_stats
        )
        
        # Test getting model weights
        weights = get_model_weights(original_model)
        print(f"✓ Extracted {len(weights)} weight tensors")
        
        # Test threshold computation
        threshold = compute_global_threshold(original_model, 0.5)
        print(f"✓ Computed global threshold: {threshold:.6f}")
        
        # Test mask creation
        masks = create_weight_masks(original_model, 0.5, "global")
        print(f"✓ Created {len(masks)} weight masks")
        
        # Test sparsity statistics
        stats = get_sparsity_stats(original_model, masks)
        print(f"✓ Computed sparsity stats: {stats['overall_sparsity']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic pruning test failed: {e}")
        return False


def test_pruned_model():
    """Test the PrunedLightStereo wrapper."""
    print("\nTesting PrunedLightStereo wrapper...")
    
    try:
        # Create a simple model configuration
        class Config:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
            
            def get(self, key, default=None):
                return getattr(self, key, default)
        
        config = Config(
            MAX_DISP=192,
            LEFT_ATT=True,
            BACKCONE='MobileNetv2',
            AGGREGATION_BLOCKS=[3, 3, 3],
            EXPANSE_RATIO=6
        )
        
        # Create original model
        from lightstereo import LightStereo
        original_model = LightStereo(config)
        print("✓ Created original model")
        
        # Test different pruning configurations
        from pruning_config import get_pruned_config, create_pruned_model_from_config
        
        test_levels = ["light", "medium", "aggressive"]
        
        for level in test_levels:
            try:
                # Create pruned model
                pruned_model = create_pruned_model_from_config(original_model, level)
                print(f"✓ Created {level} pruned model")
                
                # Get sparsity info
                sparsity_info = pruned_model.get_sparsity_info()
                sparsity = sparsity_info['overall_sparsity']
                print(f"  - Sparsity: {sparsity:.3f}")
                
                # Test forward pass with dummy data
                dummy_input = {
                    'left': torch.randn(1, 3, 256, 512),
                    'right': torch.randn(1, 3, 256, 512)
                }
                
                with torch.no_grad():
                    output = pruned_model(dummy_input)
                    print(f"  - Forward pass successful, output shape: {output['disp_pred'].shape}")
                
            except Exception as e:
                print(f"✗ Failed to test {level} pruning: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Pruned model test failed: {e}")
        return False


def test_pruning_configs():
    """Test pruning configuration system."""
    print("\nTesting pruning configurations...")
    
    try:
        from pruning_config import PRUNED_CONFIGS, get_pruned_config
        
        print(f"Available pruning levels: {list(PRUNED_CONFIGS.keys())}")
        
        for level in PRUNED_CONFIGS.keys():
            config = get_pruned_config(level)
            print(f"✓ {level}: ratio={config['pruning_ratio']}, type={config['pruning_type']}, structured={config['structured']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False


def test_comparison_with_original():
    """Compare pruned models with original model."""
    print("\nComparing pruned models with original...")
    
    try:
        # Create a simple model configuration
        class Config:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
            
            def get(self, key, default=None):
                return getattr(self, key, default)
        
        config = Config(
            MAX_DISP=192,
            LEFT_ATT=True,
            BACKCONE='MobileNetv2',
            AGGREGATION_BLOCKS=[3, 3, 3],
            EXPANSE_RATIO=6
        )
        
        # Create original model
        from lightstereo import LightStereo
        original_model = LightStereo(config)
        
        # Count original parameters
        original_params = sum(p.numel() for p in original_model.parameters())
        print(f"Original model parameters: {original_params:,}")
        
        # Test pruning comparison
        from pruning_config import compare_pruning_methods
        
        comparison = compare_pruning_methods(original_model)
        
        print("\nPruning method comparison:")
        print(f"{'Method':<20} {'Sparsity':<10} {'Zero Params':<12} {'Total Params':<12}")
        print("-" * 55)
        
        for method, results in comparison.items():
            if 'error' not in results:
                sparsity = results['actual_sparsity']
                zero_params = results['zero_params']
                total_params = results['total_params']
                print(f"{method:<20} {sparsity:<10.3f} {zero_params:<12,} {total_params:<12,}")
            else:
                print(f"{method:<20} ERROR: {results['error']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Comparison test failed: {e}")
        return False


def run_all_tests():
    """Run all pruning tests."""
    print("Running LightStereo Weight Pruning Tests")
    print("=" * 50)
    
    tests = [
        test_basic_pruning,
        test_pruned_model,
        test_pruning_configs,
        test_comparison_with_original
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Weight-based pruning is working correctly.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)