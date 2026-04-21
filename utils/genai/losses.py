import torch
import torch.nn.functional as F


def vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
):
    recon = F.binary_cross_entropy_with_logits(recon_x, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon + beta * kl
    return total, recon, kl


def gan_discriminator_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    real_labels = torch.ones_like(d_real)
    fake_labels = torch.zeros_like(d_fake)
    loss_real = F.binary_cross_entropy_with_logits(d_real, real_labels)
    loss_fake = F.binary_cross_entropy_with_logits(d_fake, fake_labels)
    return 0.5 * (loss_real + loss_fake)


def gan_generator_loss(d_fake: torch.Tensor) -> torch.Tensor:
    labels = torch.ones_like(d_fake)
    return F.binary_cross_entropy_with_logits(d_fake, labels)


def diffusion_noise_loss(pred_noise: torch.Tensor, target_noise: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_noise, target_noise)
