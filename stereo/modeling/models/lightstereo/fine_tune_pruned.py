"""
Fine-tuning script for pruned LightStereo models.
Supports progressive pruning during training with accuracy recovery.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Callable

from .advanced_pruned_lightstereo import AdvancedPrunedLightStereo, HybridPruningTrainer
from .lightstereo import LightStereo
from .pruning_utils import PruningScheduler, ModelAnalyzer


class PruningAwareFinetuner:
    """
    Fine-tuner for pruned stereo models with progressive pruning and recovery.
    """
    
    def __init__(self, 
                 original_model: LightStereo,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 device: torch.device,
                 pruning_config: Dict = None,
                 fine_tune_config: Dict = None):
        """
        Initialize the fine-tuner.
        
        Args:
            original_model: Trained LightStereo model
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            device: Device to run on
            pruning_config: Pruning configuration
            fine_tune_config: Fine-tuning hyperparameters
        """
        self.device = device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # Default fine-tuning config
        self.default_fine_tune_config = {
            'epochs': 15,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'warmup_epochs': 2,
            'scheduler_type': 'cosine',  # 'cosine', 'step', 'exponential'
            'patience': 5,
            'min_lr': 1e-6,
            'gradient_clip': 1.0,
            'progressive_pruning': True,
            'pruning_start_epoch': 3,
            'evaluation_frequency': 1
        }
        
        self.fine_tune_config = fine_tune_config if fine_tune_config else self.default_fine_tune_config
        
        # Initialize pruning trainer
        self.pruning_trainer = HybridPruningTrainer(original_model, pruning_config)
        self.model = self.pruning_trainer.create_pruned_model().to(device)
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.lr_scheduler = self._create_lr_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'sparsity': [],
            'model_size': []
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer for fine-tuning."""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.fine_tune_config['learning_rate'],
            weight_decay=self.fine_tune_config['weight_decay']
        )
    
    def _create_lr_scheduler(self):
        """Create learning rate scheduler."""
        scheduler_type = self.fine_tune_config['scheduler_type']
        
        if scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.fine_tune_config['epochs'],
                eta_min=self.fine_tune_config['min_lr']
            )
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=5,
                gamma=0.5
            )
        elif scheduler_type == 'exponential':
            return optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=0.95
            )
        else:
            return optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)
    
    def _warmup_learning_rate(self, epoch: int, step: int):
        """Apply learning rate warmup."""
        warmup_epochs = self.fine_tune_config['warmup_epochs']
        if epoch < warmup_epochs:
            base_lr = self.fine_tune_config['learning_rate']
            warmup_factor = (epoch * len(self.train_dataloader) + step) / (warmup_epochs * len(self.train_dataloader))
            current_lr = base_lr * warmup_factor
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_dataloader)
        
        for batch_idx, batch_data in enumerate(self.train_dataloader):
            # Move data to device
            for key in batch_data:
                if isinstance(batch_data[key], torch.Tensor):
                    batch_data[key] = batch_data[key].to(self.device)
            
            # Apply warmup
            self._warmup_learning_rate(epoch, batch_idx)
            
            # Forward pass
            self.optimizer.zero_grad()
            model_pred = self.model(batch_data)
            loss, _ = self.model.get_loss(model_pred, batch_data)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.fine_tune_config['gradient_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.fine_tune_config['gradient_clip']
                )
            
            self.optimizer.step()
            
            # Progressive pruning
            if (self.fine_tune_config['progressive_pruning'] and 
                epoch >= self.fine_tune_config['pruning_start_epoch']):
                global_step = epoch * num_batches + batch_idx
                self.pruning_trainer.progressive_pruning_step(global_step)
            
            total_loss += loss.item()
            
            # Log progress
            if batch_idx % 50 == 0:
                self.logger.info(
                    f'Epoch {epoch}, Batch {batch_idx}/{num_batches}, '
                    f'Loss: {loss.item():.4f}, LR: {self.optimizer.param_groups[0]["lr"]:.2e}'
                )
        
        return total_loss / num_batches
    
    def validate(self) -> Tuple[float, Dict]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        metrics = {'disparity_error': 0.0, 'bad_pixels': 0.0}
        num_batches = len(self.val_dataloader)
        
        with torch.no_grad():
            for batch_data in self.val_dataloader:
                # Move data to device
                for key in batch_data:
                    if isinstance(batch_data[key], torch.Tensor):
                        batch_data[key] = batch_data[key].to(self.device)
                
                # Forward pass
                model_pred = self.model(batch_data)
                loss, _ = self.model.get_loss(model_pred, batch_data)
                total_loss += loss.item()
                
                # Calculate stereo-specific metrics
                if 'disp' in batch_data:
                    disp_pred = model_pred['disp_pred']
                    disp_gt = batch_data['disp']
                    
                    # Calculate disparity error
                    mask = (disp_gt > 0) & (disp_gt < self.model.max_disp)
                    if mask.sum() > 0:
                        error = torch.abs(disp_pred[mask] - disp_gt[mask])
                        metrics['disparity_error'] += error.mean().item()
                        
                        # Bad pixel ratio (error > 3 pixels)
                        bad_pixels = (error > 3.0).float().mean()
                        metrics['bad_pixels'] += bad_pixels.item()
        
        avg_loss = total_loss / num_batches
        for key in metrics:
            metrics[key] /= num_batches
        
        return avg_loss, metrics
    
    def fine_tune(self) -> Dict:
        """
        Execute the complete fine-tuning process.
        
        Returns:
            Training history and final model statistics
        """
        self.logger.info("Starting fine-tuning of pruned LightStereo model...")
        self.logger.info(f"Training for {self.fine_tune_config['epochs']} epochs")
        
        # Apply initial structured pruning
        if not self.model.is_pruned:
            self.model.apply_structured_pruning()
        
        for epoch in range(self.fine_tune_config['epochs']):
            self.current_epoch = epoch
            
            # Train epoch
            train_loss = self.train_epoch(epoch)
            
            # Validate
            if epoch % self.fine_tune_config['evaluation_frequency'] == 0:
                val_loss, val_metrics = self.validate()
                
                # Update learning rate scheduler
                if hasattr(self.lr_scheduler, 'step'):
                    self.lr_scheduler.step()
                
                # Record history
                self.training_history['train_loss'].append(train_loss)
                self.training_history['val_loss'].append(val_loss)
                self.training_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
                
                # Model statistics
                sparsity = ModelAnalyzer.calculate_sparsity(self.model.model)
                model_size = self.model.get_model_size()[1]
                self.training_history['sparsity'].append(sparsity)
                self.training_history['model_size'].append(model_size)
                
                # Log epoch results
                self.logger.info(
                    f'Epoch {epoch}: Train Loss: {train_loss:.4f}, '
                    f'Val Loss: {val_loss:.4f}, '
                    f'Disp Error: {val_metrics["disparity_error"]:.3f}, '
                    f'Bad Pixels: {val_metrics["bad_pixels"]:.1%}, '
                    f'Sparsity: {sparsity:.1%}'
                )
                
                # Early stopping check
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self._save_best_model()
                else:
                    self.patience_counter += 1
                
                if self.patience_counter >= self.fine_tune_config['patience']:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        # Apply final full pruning if not done progressively
        if not self.fine_tune_config['progressive_pruning']:
            self.logger.info("Applying final hybrid pruning...")
            self.pruning_trainer.apply_full_hybrid_pruning()
        
        # Final evaluation
        final_val_loss, final_metrics = self.validate()
        
        self.logger.info("Fine-tuning completed!")
        self._print_final_statistics(final_val_loss, final_metrics)
        
        return {
            'training_history': self.training_history,
            'final_metrics': final_metrics,
            'final_val_loss': final_val_loss,
            'best_val_loss': self.best_val_loss,
            'model': self.model
        }
    
    def _save_best_model(self):
        """Save the best model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history
        }
        torch.save(checkpoint, 'best_pruned_lightstereo.pth')
    
    def _print_final_statistics(self, final_val_loss: float, final_metrics: Dict):
        """Print final training statistics."""
        original_params = ModelAnalyzer.count_parameters(self.pruning_trainer.original_model)
        pruned_params = ModelAnalyzer.count_parameters(self.model.model)
        compression_ratio = original_params / pruned_params if pruned_params > 0 else float('inf')
        sparsity = ModelAnalyzer.calculate_sparsity(self.model.model)
        
        print("\n" + "="*50)
        print("FINE-TUNING RESULTS")
        print("="*50)
        print(f"Final validation loss: {final_val_loss:.4f}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Final disparity error: {final_metrics['disparity_error']:.3f}")
        print(f"Final bad pixel ratio: {final_metrics['bad_pixels']:.1%}")
        print(f"Model compression: {compression_ratio:.2f}x")
        print(f"Final sparsity: {sparsity:.1%}")
        print(f"Model size: {self.model.get_model_size()[1]:.1f} MB")
        print("="*50)


def create_fine_tune_config(
    epochs: int = 15,
    learning_rate: float = 1e-4,
    progressive_pruning: bool = True,
    **kwargs
) -> Dict:
    """
    Create fine-tuning configuration.
    
    Args:
        epochs: Number of training epochs
        learning_rate: Initial learning rate
        progressive_pruning: Enable progressive pruning during training
        **kwargs: Additional configuration options
    
    Returns:
        Configuration dictionary
    """
    config = {
        'epochs': epochs,
        'learning_rate': learning_rate,
        'weight_decay': kwargs.get('weight_decay', 1e-5),
        'warmup_epochs': kwargs.get('warmup_epochs', 2),
        'scheduler_type': kwargs.get('scheduler_type', 'cosine'),
        'patience': kwargs.get('patience', 5),
        'min_lr': kwargs.get('min_lr', 1e-6),
        'gradient_clip': kwargs.get('gradient_clip', 1.0),
        'progressive_pruning': progressive_pruning,
        'pruning_start_epoch': kwargs.get('pruning_start_epoch', 3),
        'evaluation_frequency': kwargs.get('evaluation_frequency', 1)
    }
    
    return config


def fine_tune_pruned_lightstereo(
    original_model: LightStereo,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    device: torch.device,
    pruning_config: Dict = None,
    fine_tune_config: Dict = None
) -> Dict:
    """
    Convenience function to fine-tune a pruned LightStereo model.
    
    Args:
        original_model: Trained LightStereo model
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        device: Device to run on
        pruning_config: Pruning configuration
        fine_tune_config: Fine-tuning configuration
    
    Returns:
        Training results and final model
    """
    finetuner = PruningAwareFinetuner(
        original_model=original_model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        pruning_config=pruning_config,
        fine_tune_config=fine_tune_config
    )
    
    return finetuner.fine_tune()