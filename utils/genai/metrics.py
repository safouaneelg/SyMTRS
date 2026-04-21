import torch
import torch.nn.functional as F


def reconstruction_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def gan_discriminator_accuracy(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    # Works for logits (default) and probabilities.
    if d_real.min() < 0 or d_real.max() > 1 or d_fake.min() < 0 or d_fake.max() > 1:
        real_ok = (d_real >= 0.0).float().mean()
        fake_ok = (d_fake < 0.0).float().mean()
    else:
        real_ok = (d_real >= 0.5).float().mean()
        fake_ok = (d_fake < 0.5).float().mean()
    return 0.5 * (real_ok + fake_ok)
