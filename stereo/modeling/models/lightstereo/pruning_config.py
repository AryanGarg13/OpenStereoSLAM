# Configuration for Weight-Based Pruned LightStereo Model
# This configuration uses magnitude-based pruning to zero out smallest weights

import yaml

# Weight-based pruning configurations
PRUNED_CONFIGS = {
    "light": {
        "pruning_ratio": 0.3,  # Light pruning - 30% of weights pruned
        "pruning_type": "global",  # Global magnitude-based pruning
        "structured": False,   # Unstructured pruning
        "description": "Light magnitude-based pruning for minimal performance impact"
    },
    
    "medium": {
        "pruning_ratio": 0.5,  # Medium pruning - 50% of weights pruned
        "pruning_type": "global",  # Global magnitude-based pruning
        "structured": False,   # Unstructured pruning
        "description": "Balanced magnitude-based pruning for good performance/efficiency trade-off"
    },
    
    "aggressive": {
        "pruning_ratio": 0.7,  # Aggressive pruning - 70% of weights pruned
        "pruning_type": "global",  # Global magnitude-based pruning
        "structured": False,   # Unstructured pruning
        "description": "Aggressive magnitude-based pruning for maximum efficiency"
    },
    
    "structured_light": {
        "pruning_ratio": 0.25,  # Light structured pruning
        "pruning_type": "global",
        "structured": True,    # Structured channel pruning
        "description": "Light structured pruning - removes entire channels"
    },
    
    "structured_medium": {
        "pruning_ratio": 0.4,   # Medium structured pruning
        "pruning_type": "global",
        "structured": True,    # Structured channel pruning
        "description": "Medium structured pruning - removes entire channels"
    },
    
    "layerwise_medium": {
        "pruning_ratio": 0.5,   # Medium layer-wise pruning
        "pruning_type": "layerwise",  # Per-layer pruning
        "structured": False,
        "description": "Layer-wise magnitude-based pruning"
    }
}


def get_pruned_config(pruning_level="medium"):
    """
    Get pruning configuration for weight-based pruning.
    
    Args:
        pruning_level: One of the pruning levels defined in PRUNED_CONFIGS
        
    Returns:
        Dictionary with pruning configuration parameters
    """
    if pruning_level not in PRUNED_CONFIGS:
        available_levels = list(PRUNED_CONFIGS.keys())
        raise ValueError(f"Invalid pruning level: {pruning_level}. Choose from {available_levels}")
    
    return PRUNED_CONFIGS[pruning_level].copy()


def create_pruned_model_from_config(base_model, pruning_level="medium"):
    """
    Create a pruned model using configuration.
    
    Args:
        base_model: Original LightStereo model
        pruning_level: Pruning level configuration to use
        
    Returns:
        PrunedLightStereo model
    """
    from .pruned_lightstereo import PrunedLightStereo
    
    pruning_config = get_pruned_config(pruning_level)
    return PrunedLightStereo(base_model, pruning_config)


def create_pruned_config_file(output_path, pruning_level="medium"):
    """
    Create a YAML configuration file for weight-based pruned model.
    
    Args:
        output_path: Path to save the pruned configuration
        pruning_level: Pruning intensity level
    """
    pruning_config = get_pruned_config(pruning_level)
    
    with open(output_path, 'w') as f:
        yaml.dump(pruning_config, f, default_flow_style=False, indent=2)
    
    print(f"Pruning configuration saved to: {output_path}")
    print(f"Pruning level: {pruning_level}")
    print(f"Pruning ratio: {pruning_config['pruning_ratio']}")
    print(f"Pruning type: {pruning_config['pruning_type']}")
    print(f"Structured: {pruning_config['structured']}")


def compare_pruning_methods(base_model):
    """
    Compare different pruning methods on a base model.
    
    Args:
        base_model: Original LightStereo model
        
    Returns:
        Dictionary with comparison results
    """
    try:
        from .weight_pruning import get_sparsity_stats, create_weight_masks
        
        results = {}
        
        # Test different pruning configurations
        test_configs = ["light", "medium", "aggressive", "structured_medium", "layerwise_medium"]
        
        for config_name in test_configs:
            try:
                config = get_pruned_config(config_name)
                
                if config['structured']:
                    from .weight_pruning import structured_channel_pruning
                    masks = structured_channel_pruning(base_model, config['pruning_ratio'])
                else:
                    masks = create_weight_masks(base_model, config['pruning_ratio'], config['pruning_type'])
                
                stats = get_sparsity_stats(base_model, masks)
                
                results[config_name] = {
                    'config': config,
                    'actual_sparsity': stats['overall_sparsity'],
                    'total_params': stats['total_params'],
                    'zero_params': stats['zero_params']
                }
                
            except Exception as e:
                results[config_name] = {'error': str(e)}
        
        return results
        
    except ImportError as e:
        return {'error': f'Required dependencies not available: {e}'}


if __name__ == "__main__":
    # Example: Create pruned configuration files
    import os
    
    output_dir = "./pruned_configs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create configuration files for all pruning levels
    for level in PRUNED_CONFIGS.keys():
        output_file = os.path.join(output_dir, f"lightstereo_pruned_{level}.yaml")
        create_pruned_config_file(output_file, pruning_level=level)