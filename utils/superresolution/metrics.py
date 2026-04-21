# E:\utils\superresolution\metrics.py
import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return torch.tensor(100.0, device=pred.device)
    return 20.0 * torch.log10(max_val / torch.sqrt(mse))


def _gaussian_window(window_size: int, sigma: float, channels: int, device):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_1d = g.reshape(1, 1, 1, -1)
    window_2d = window_1d.transpose(2, 3) @ window_1d
    window = window_2d.expand(channels, 1, window_size, window_size)
    return window


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_val: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    # pred/target: [B,C,H,W] in [0,1]
    h, w = pred.shape[2], pred.shape[3]
    if h < window_size or w < window_size:
        window_size = min(window_size, h, w)
    if window_size < 3:
        return torch.tensor(0.0, device=pred.device)
    if window_size % 2 == 0:
        window_size -= 1
    channels = pred.shape[1]
    window = _gaussian_window(window_size, sigma, channels, pred.device)

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channels)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=channels) - mu1_mu2

    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()
