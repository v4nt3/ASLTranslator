import numpy as np

path = "data/features_fused/66221823911-CARRY_clip0_fused.npy"

data = np.load(path)
print("Shape:", data.shape)
print("Dtype:", data.dtype)
