from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
import numpy as np

def batch_ssim(a,b):
    ssim_values = []
    
    if not isinstance(b, np.ndarray):
        b = b.numpy()

    for a, b in zip(a, b):
        ssim_val = structural_similarity(a, b, data_range = 1)
        ssim_values.append(ssim_val)

    return np.mean(ssim_values)

def batch_ssim_full(a,b):
    ssim_values = []
    ssim_maps = []
    
    if not isinstance(b, np.ndarray):
        b = b.numpy()

    for a, b in zip(a, b):
        ssim_val, ssim_map = structural_similarity(a, b, full = True, data_range = 1)
        ssim_values.append(ssim_val)
        ssim_maps.append(ssim_map)

    return np.mean(ssim_values), ssim_maps

def batch_psnr(a,b):
    psnr_values = []
    if not isinstance(b, np.ndarray):
        b = b.numpy()

    for a, b in zip(a, b):
        psnr_val = peak_signal_noise_ratio(a, b, data_range = 1)
        psnr_values.append(psnr_val)

    return np.mean(psnr_values)

def batch_mse(y_true, y_pred):
    # y_true = np.array(y_true)
    # y_pred = np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)

def batch_rmse(y_true, y_pred):
    return np.sqrt(batch_mse(y_true, y_pred))

def batch_mae(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))
