"""
Example usage of the hybrid pruned LightStereo model.
Demonstrates how to prune, fine-tune, and evaluate the model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import your existing LightStereo model and data loading utilities
from .lightstereo import LightStereo
from .advanced_pruned_lightstereo import create_hybrid_pruned_lightstereo
from .fine_tune_pruned import fine_tune_pruned_lightstereo, create_fine_tune_config
from .evaluate_pruned import evaluate_pruned_lightstereo


def example_pruning_pipeline():
    """
    Complete example of the pruning pipeline.
    """
    
    # 1. Load your pre-trained LightStereo model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Assuming you have a configuration for LightStereo
    lightstereo_config = {
        'MAX_DISP': 192,
        'LEFT_ATT': False,
        'AGGREGATION_BLOCKS': 8,
        'EXPANSE_RATIO': 2
    }
    
    # Load pre-trained model (replace with your actual model loading)
    original_model = LightStereo(lightstereo_config)
    # original_model.load_state_dict(torch.load('your_pretrained_weights.pth'))
    original_model.to(device)
    
    print("Loaded original LightStereo model")
    
    # 2. Create pruned model with custom configuration
    pruning_config = {
        'unstructured_ratios': {
            'backbone': 0.4,        # 40% unstructured pruning on backbone
            'cost_agg': 0.3,        # 30% on cost aggregation
            'refine_1': 0.5,        # 50% on refinement layers
        },
        'structured_ratios': {
            'backbone_channels': 0.25,      # Remove 25% of backbone channels
            'cost_agg_channels': 0.15,      # Remove 15% of cost_agg channels
            'refine_2_channels': 0.3,       # Remove 30% of refine_2 channels
            'refine_3_channels': 0.4,       # Remove 40% of refine_3 channels
        },
        'importance_metric': 'l1',          # Use L1 norm for channel importance
        'exclude_layers': ['stem_2']        # Don't prune input processing
    }
    
    # Create pruned model
    pruned_model = create_hybrid_pruned_lightstereo(
        original_model=original_model,
        unstructured_ratios=pruning_config['unstructured_ratios'],
        structured_ratios=pruning_config['structured_ratios']
    )
    
    print("Created hybrid pruned model")
    
    # 3. Apply pruning
    pruned_model.apply_full_pruning()
    
    print("Applied hybrid pruning (60% unstructured + 40% structured)")
    
    # 4. Setup data loaders (replace with your actual data loading)
    # train_dataloader = DataLoader(...)  # Your training dataset
    # val_dataloader = DataLoader(...)    # Your validation dataset
    
    # For demonstration, create dummy data loaders
    class DummyDataset:
        def __len__(self):
            return 100
        
        def __getitem__(self, idx):
            return {
                'left': torch.randn(3, 480, 640),
                'right': torch.randn(3, 480, 640),
                'disp': torch.randn(480, 640) * 50 + 10  # Dummy disparity
            }
    
    train_dataloader = DataLoader(DummyDataset(), batch_size=4, shuffle=True)
    val_dataloader = DataLoader(DummyDataset(), batch_size=4, shuffle=False)
    
    # 5. Fine-tune the pruned model
    fine_tune_config = create_fine_tune_config(
        epochs=15,
        learning_rate=1e-4,
        progressive_pruning=True,
        weight_decay=1e-5,
        warmup_epochs=2
    )
    
    print("Starting fine-tuning...")
    
    fine_tune_results = fine_tune_pruned_lightstereo(
        original_model=original_model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        pruning_config=pruning_config,
        fine_tune_config=fine_tune_config
    )
    
    fine_tuned_model = fine_tune_results['model']
    
    print("Fine-tuning completed!")
    print(f"Best validation loss: {fine_tune_results['best_val_loss']:.4f}")
    
    # 6. Evaluate the pruned model
    print("Evaluating pruned model...")
    
    evaluation_results = evaluate_pruned_lightstereo(
        original_model=original_model,
        pruned_model=fine_tuned_model,
        dataloader=val_dataloader,
        device=device,
        generate_report=True,
        save_plots='pruning_comparison.png'
    )
    
    # 7. Save the pruned model
    fine_tuned_model.save_pruned_model('pruned_lightstereo_final.pth')
    
    print("Pruned model saved!")
    
    return {
        'original_model': original_model,
        'pruned_model': fine_tuned_model,
        'fine_tune_results': fine_tune_results,
        'evaluation_results': evaluation_results
    }


def quick_pruning_example():
    """
    Quick example with default settings.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model (replace with actual loading)
    lightstereo_config = {'MAX_DISP': 192, 'LEFT_ATT': False, 'AGGREGATION_BLOCKS': 8, 'EXPANSE_RATIO': 2}
    original_model = LightStereo(lightstereo_config).to(device)
    
    # Create pruned model with defaults (60% unstructured + 40% structured)
    pruned_model = create_hybrid_pruned_lightstereo(original_model)
    
    # Apply pruning
    pruned_model.apply_full_pruning()
    
    # Get model statistics
    original_params, original_size = original_model.numel(), sum(p.numel() for p in original_model.parameters()) * 4 / (1024**2)
    pruned_params, pruned_size = pruned_model.get_model_size()
    
    print(f"Original model: {original_params:,} parameters, {original_size:.1f} MB")
    print(f"Pruned model: {pruned_params:,} parameters, {pruned_size:.1f} MB")
    print(f"Compression ratio: {original_params / pruned_params:.2f}x")
    
    return pruned_model


def load_and_use_pruned_model():
    """
    Example of loading a saved pruned model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load original model for reference
    lightstereo_config = {'MAX_DISP': 192, 'LEFT_ATT': False, 'AGGREGATION_BLOCKS': 8, 'EXPANSE_RATIO': 2}
    original_model = LightStereo(lightstereo_config).to(device)
    
    # Load pruned model
    try:
        from .advanced_pruned_lightstereo import AdvancedPrunedLightStereo
        pruned_model = AdvancedPrunedLightStereo.load_pruned_model(
            'pruned_lightstereo_final.pth', 
            original_model
        ).to(device)
        
        print("Loaded pruned model successfully!")
        
        # Use the model for inference
        dummy_data = {
            'left': torch.randn(1, 3, 480, 640, device=device),
            'right': torch.randn(1, 3, 480, 640, device=device)
        }
        
        with torch.no_grad():
            result = pruned_model(dummy_data)
            print(f"Inference successful! Output shape: {result['disp_pred'].shape}")
        
        return pruned_model
        
    except FileNotFoundError:
        print("Saved pruned model not found. Run example_pruning_pipeline() first.")
        return None


def benchmark_comparison():
    """
    Benchmark original vs pruned model performance.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create models
    lightstereo_config = {'MAX_DISP': 192, 'LEFT_ATT': False, 'AGGREGATION_BLOCKS': 8, 'EXPANSE_RATIO': 2}
    original_model = LightStereo(lightstereo_config).to(device)
    
    pruned_model = create_hybrid_pruned_lightstereo(original_model)
    pruned_model.apply_full_pruning()
    
    # Benchmark inference time
    import time
    
    dummy_data = {
        'left': torch.randn(1, 3, 480, 640, device=device),
        'right': torch.randn(1, 3, 480, 640, device=device)
    }
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = original_model(dummy_data)
            _ = pruned_model(dummy_data)
    
    # Benchmark original
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    for _ in range(100):
        with torch.no_grad():
            _ = original_model(dummy_data)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    original_time = (time.time() - start_time) / 100
    
    # Benchmark pruned
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    for _ in range(100):
        with torch.no_grad():
            _ = pruned_model(dummy_data)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    pruned_time = (time.time() - start_time) / 100
    
    print(f"Original model: {original_time*1000:.1f} ms")
    print(f"Pruned model: {pruned_time*1000:.1f} ms")
    print(f"Speedup: {original_time/pruned_time:.2f}x")


if __name__ == "__main__":
    print("Running hybrid pruning example...")
    
    # Run quick example
    print("\n1. Quick pruning example:")
    quick_pruned = quick_pruning_example()
    
    # Run benchmark comparison
    print("\n2. Performance benchmark:")
    benchmark_comparison()
    
    # Uncomment to run full pipeline (requires actual training data)
    # print("\n3. Full pruning pipeline:")
    # results = example_pruning_pipeline()
    
    print("\nExample completed!")