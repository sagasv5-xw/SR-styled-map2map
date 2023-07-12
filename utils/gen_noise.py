import numpy as np
import torch


size_list = [64, 128, 256, 512]
seed = 42
noise_saving_path = '/hildafs/home/xzhangn/xzhangn/sr_pipeline/4-postproc/SR-styled-map2map/noise/noise_{}.npy'

for size in size_list:
    if seed is not None:
        torch.manual_seed(seed)
    noise = torch.randn(1, size, size, size)
    noise_numpy = noise.cpu().numpy()
    np.save(noise_saving_path.format(size), noise_numpy)