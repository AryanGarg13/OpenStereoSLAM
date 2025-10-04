# @Time    : 2024/2/9 11:39
# @Author  : zhangchenming
import torch
from stereo.modeling.trainer_template import TrainerTemplate
from .lightstereo import LightStereo
from .advanced_pruned_lightstereo import AdvancedPrunedLightStereo, create_hybrid_pruned_lightstereo

__all__ = {
    'LightStereo': LightStereo,
    'PrunedLightStereo': AdvancedPrunedLightStereo,
}


class Trainer(TrainerTemplate):
    def __init__(self, args, cfgs, local_rank, global_rank, logger, tb_writer):
        model_name = cfgs.MODEL.NAME

        # Check if we're training a pruned model
        if model_name == 'PrunedLightStereo':
            # Load the original pretrained model first
            pretrained_path = cfgs.MODEL.get('PRETRAINED_MODEL', '')
            if not pretrained_path:
                raise ValueError("PRETRAINED_MODEL path is required for PrunedLightStereo training")

            logger.info(f"Loading pretrained model from: {pretrained_path}")

            # Create original LightStereo model
            original_model = LightStereo(cfgs.MODEL)

            # Load pretrained weights
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                original_model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                original_model.load_state_dict(checkpoint['state_dict'])
            else:
                original_model.load_state_dict(checkpoint)

            logger.info("Pretrained model loaded successfully")

            # Get pruning configuration from model config
            pruning_config = cfgs.MODEL.get('PRUNING_CONFIG', None)

            # Create pruned model
            logger.info("Creating pruned model...")
            model = AdvancedPrunedLightStereo(original_model, pruning_config)

            # Apply pruning if specified
            if cfgs.MODEL.get('APPLY_PRUNING_ON_INIT', True):
                logger.info("Applying hybrid pruning (60% unstructured + 40% structured)...")
                global_sparsity = cfgs.MODEL.get('GLOBAL_SPARSITY', None)
                model.apply_full_pruning(global_sparsity)
                logger.info("Pruning applied successfully")

                # Print model statistics
                params, size_mb = model.get_model_size()
                logger.info(f"Pruned model size: {params:,} parameters ({size_mb:.2f} MB)")
        else:
            # Standard model training
            model = __all__[model_name](cfgs.MODEL)

        super().__init__(args, cfgs, local_rank, global_rank, logger, tb_writer, model)
