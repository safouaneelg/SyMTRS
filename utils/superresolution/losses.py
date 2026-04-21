# E:\utils\superresolution\losses.py
import torch
import torch.nn.functional as F


def pixel_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def srgan_discriminator_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    # Relativistic average GAN style not used; simple BCEWithLogits
    real_labels = torch.ones_like(d_real)
    fake_labels = torch.zeros_like(d_fake)
    loss_real = F.binary_cross_entropy_with_logits(d_real, real_labels)
    loss_fake = F.binary_cross_entropy_with_logits(d_fake, fake_labels)
    return (loss_real + loss_fake) * 0.5


def srgan_generator_loss(
    d_fake: torch.Tensor,
    sr: torch.Tensor,
    hr: torch.Tensor,
    adv_weight: float = 1e-3,
) -> torch.Tensor:
    adv_labels = torch.ones_like(d_fake)
    adv_loss = F.binary_cross_entropy_with_logits(d_fake, adv_labels)
    content_loss = F.mse_loss(sr, hr)
    return content_loss + adv_weight * adv_loss
