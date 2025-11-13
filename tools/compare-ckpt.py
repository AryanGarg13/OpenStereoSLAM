import torch
import os

# Paths to checkpoints
fp32_path = "/home/aniruth/Desktop/Courses/Independent - Study/OpenStereoSLAM/output/KittiDataset/LightStereo/lightstereo_m_kitti/default/ckpt/checkpoint_epoch_0.pth"
quant_path = "/home/aniruth/Desktop/Courses/Independent - Study/OpenStereoSLAM/output/KittiDataset/LightStereo/lightstereo_m_kitti/default/ckpt/model_quantized_final.pth"

# Load checkpoints
fp32_ckpt = torch.load(fp32_path, map_location="cpu")
quant_ckpt = torch.load(quant_path, map_location="cpu")

# FP32 model_state
fp32_state = fp32_ckpt['model_state']

# Function to count parameters by dtype
def count_params_dtype(state_dict):
    dtype_count = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            dtype = str(v.dtype)
            dtype_count[dtype] = dtype_count.get(dtype, 0) + v.numel()
    return dtype_count

# FP32
fp32_dtype_count = count_params_dtype(fp32_state)

# Quantized
# Treat 'weight' tensors as int8 (others like bias/metadata as float32)
int8_params = 0
float32_params = 0
for k, v in quant_ckpt.items():
    if 'weight' in k:
        int8_params += v.numel()
    elif isinstance(v, torch.Tensor) and v.dtype == torch.float32:
        float32_params += v.numel()

# Get checkpoint sizes on disk (MB)
fp32_size = os.path.getsize(fp32_path) / (1024*1024)
quant_size = os.path.getsize(quant_path) / (1024*1024)

# Print results
print(f"FP32 checkpoint size: {fp32_size:.2f} MB")
print("FP32 parameter counts by dtype:")
for dt, count in fp32_dtype_count.items():
    print(f"{dt}: {count}")

print("\nQuantized checkpoint size: {:.2f} MB".format(quant_size))
print("Quantized model parameter counts:")
print(f"int8 (weights): {int8_params}")
print(f"float32 (bias/metadata): {float32_params}")
