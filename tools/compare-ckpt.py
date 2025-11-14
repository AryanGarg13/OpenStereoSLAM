import torch
import os

fp32_path = "/home/aniruth/Desktop/Courses/Independent - Study/OpenStereoSLAM/output/KittiDataset/LightStereo/lightstereo_m_kitti/default-baseline/ckpt/checkpoint_best.pth"
quant_path = "/home/aniruth/Desktop/Courses/Independent - Study/OpenStereoSLAM/output/KittiDataset/LightStereo/lightstereo_m_kitti/default/ckpt/checkpoint_quantized_final.pth"

fp32_ckpt = torch.load(fp32_path, map_location="cpu")
quant_ckpt = torch.load(quant_path, map_location="cpu")

print("FP32 Checkpoint keys:", fp32_ckpt.keys())
print("Quantized Checkpoint keys:", quant_ckpt.keys())

fp32_state = fp32_ckpt['model_state']
quant_state = quant_ckpt['model_state']

def count_params_dtype(state_dict, quant=False):
    int8_params = 0
    float32_params = 0
    total_bytes = 0
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            if quant and 'weight' in k:
                int8_params += v.numel()
                total_bytes += v.numel()  # int8 = 1 byte
            else:
                float32_params += v.numel()
                total_bytes += v.numel() * 4  # float32 = 4 bytes
    return int8_params, float32_params, total_bytes

# FP32
fp32_int8, fp32_float32, fp32_bytes = count_params_dtype(fp32_state)
# Quantized
quant_int8, quant_float32, quant_bytes = count_params_dtype(quant_state, quant=True)

print("FP32 model size (in RAM, model_state only): {:.2f} MB".format(fp32_bytes / (1024*1024)))
print("FP32 parameter counts:")
print(f"int8 (weights): {fp32_int8}  # usually 0")
print(f"float32 (bias/weights/others): {fp32_float32}")

print("\nQuantized model size (in RAM, model_state only): {:.2f} MB".format(quant_bytes / (1024*1024)))
print("Quantized parameter counts:")
print(f"int8 (weights): {quant_int8}")
print(f"float32 (bias/metadata): {quant_float32}")

