# E:\train_superresolution.py
import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

# Local imports
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))
sys.path.append(r"E:\\dataloader")

from dataloader import SuperResDataset, split_indices
from models.superresolution import SRCNN, AutoencoderSR, SRGANGenerator, SRGANDiscriminator, SwinIR
from utils.superresolution import (
    AverageMeter,
    match_size,
    pixel_mse_loss,
    psnr,
    ssim,
    rgb_to_y,
    save_checkpoint,
    save_image_triplet,
    set_seed,
    srgan_discriminator_loss,
    srgan_generator_loss,
    write_csv_row,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train super-resolution models")
    p.add_argument("--lr_root", type=str, required=True, help="LR images root")
    p.add_argument("--hr_root", type=str, required=True, help="HR images root")
    p.add_argument("--model", type=str, default="srcnn", choices=["srcnn", "autoencoder", "srgan", "swinir"])
    p.add_argument("--scale", type=int, default=4, help="Upscale factor for SRGAN/SwinIR")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", type=float, nargs="+", default=[0.8, 0.2])
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--use_y_channel", action="store_true", help="Use Y channel (SRCNN style)")
    p.add_argument("--out_dir", type=str, default=str(ROOT / "sr_runs"))
    p.add_argument("--save_samples_every", type=int, default=1)
    p.add_argument("--adv_weight", type=float, default=1e-3, help="SRGAN adversarial loss weight")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--weights", type=str, default="", help="Path to checkpoint to resume from")
    p.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use (1 for single-GPU, >1 for DDP)")
    p.add_argument("--patch_size", type=int, default=0, help="LR patch size for SwinIR training (0 = disabled)")
    p.add_argument("--val_patch_size", type=int, default=0, help="LR patch size for SwinIR validation (0 = use train patch size)")
    p.add_argument("--amp", action="store_true", help="Enable AMP (SwinIR only)")
    p.add_argument("--swinir_checkpoint", action="store_true", help="Enable gradient checkpointing (SwinIR only)")
    # SwinIR config (optional overrides)
    p.add_argument("--swinir_embed_dim", type=int, default=96)
    p.add_argument("--swinir_depths", type=int, nargs="+", default=[6, 6, 6, 6])
    p.add_argument("--swinir_num_heads", type=int, nargs="+", default=[6, 6, 6, 6])
    p.add_argument("--swinir_window", type=int, default=7)
    p.add_argument("--swinir_mlp_ratio", type=float, default=4.0)
    p.add_argument("--swinir_upsampler", type=str, default="pixelshuffle")
    return p.parse_args()


def build_models(args, in_channels, out_channels, img_size=None):
    if args.model == "srcnn":
        return SRCNN(in_channels=in_channels, out_channels=out_channels), None
    if args.model == "autoencoder":
        return AutoencoderSR(in_channels=in_channels), None
    if args.model == "srgan":
        g = SRGANGenerator(in_channels=in_channels, out_channels=out_channels, scale=args.scale)
        d = SRGANDiscriminator(in_channels=out_channels)
        return g, d
    if args.model == "swinir":
        if img_size is None:
            raise ValueError("img_size is required for SwinIR")
        g = SwinIR(
            img_size=img_size,
            in_chans=in_channels,
            embed_dim=args.swinir_embed_dim,
            depths=args.swinir_depths,
            num_heads=args.swinir_num_heads,
            window_size=args.swinir_window,
            mlp_ratio=args.swinir_mlp_ratio,
            upscale=args.scale,
            img_range=1.0,
            upsampler=args.swinir_upsampler,
            use_checkpoint=args.swinir_checkpoint,
        )
        return g, None
    raise ValueError("Unknown model")


def prepare_batch(lr, hr, model_name, use_y):
    if use_y:
        lr = rgb_to_y(lr)
        hr = rgb_to_y(hr)
    if model_name in {"srcnn", "autoencoder"}:
        lr = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False)
    return lr, hr


def collate_same_size(batch):
    if not batch:
        return None
    ref = batch[0]["lr"].shape
    filtered = [b for b in batch if b["lr"].shape == ref and b["hr"].shape == batch[0]["hr"].shape]
    if len(filtered) < len(batch):
        skipped = len(batch) - len(filtered)
        print(f"[collate] Skipping {skipped} sample(s) due to mismatched sizes in batch")
    if not filtered:
        return None
    return torch.utils.data.default_collate(filtered)


def random_crop_pair(lr, hr, scale, patch_size):
    # lr/hr: BxCxHxW
    if patch_size <= 0:
        return lr, hr
    b, _, h_lr, w_lr = lr.shape
    h_hr, w_hr = hr.shape[2], hr.shape[3]
    if h_lr < patch_size or w_lr < patch_size:
        return lr, hr
    ps_hr = patch_size * scale
    if h_hr < ps_hr or w_hr < ps_hr:
        return lr, hr

    cropped_lr = []
    cropped_hr = []
    for i in range(b):
        top = torch.randint(0, h_lr - patch_size + 1, (1,), device=lr.device).item()
        left = torch.randint(0, w_lr - patch_size + 1, (1,), device=lr.device).item()
        lr_i = lr[i : i + 1, :, top : top + patch_size, left : left + patch_size]
        hr_top = top * scale
        hr_left = left * scale
        hr_i = hr[i : i + 1, :, hr_top : hr_top + ps_hr, hr_left : hr_left + ps_hr]
        cropped_lr.append(lr_i)
        cropped_hr.append(hr_i)
    return torch.cat(cropped_lr, dim=0), torch.cat(cropped_hr, dim=0)


def center_crop_pair(lr, hr, scale, patch_size):
    if patch_size <= 0:
        return lr, hr
    b, _, h_lr, w_lr = lr.shape
    h_hr, w_hr = hr.shape[2], hr.shape[3]
    ps_hr = patch_size * scale
    if h_lr < patch_size or w_lr < patch_size or h_hr < ps_hr or w_hr < ps_hr:
        return lr, hr
    top = (h_lr - patch_size) // 2
    left = (w_lr - patch_size) // 2
    lr_c = lr[:, :, top : top + patch_size, left : left + patch_size]
    hr_top = top * scale
    hr_left = left * scale
    hr_c = hr[:, :, hr_top : hr_top + ps_hr, hr_left : hr_left + ps_hr]
    return lr_c, hr_c


def train_one_epoch(args, model, disc, loader, optim_g, optim_d, device, scaler=None):
    model.train()
    if disc is not None:
        disc.train()

    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    mse_meter = AverageMeter()
    ssim_meter = AverageMeter()

    for batch in tqdm(loader, desc="train", leave=False, disable=not args.is_main):
        if batch is None:
            continue
        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)
        lr, hr = prepare_batch(lr, hr, args.model, args.use_y_channel)
        if args.model == "swinir" and args.patch_size > 0:
            lr, hr = random_crop_pair(lr, hr, args.scale, args.patch_size)

        if args.model == "srgan":
            # Train discriminator
            with torch.no_grad():
                sr = model(lr)
            hr_c, sr_c = match_size(hr, sr)
            d_real = disc(hr_c)
            d_fake = disc(sr_c.detach())
            d_loss = srgan_discriminator_loss(d_real, d_fake)
            optim_d.zero_grad()
            d_loss.backward()
            optim_d.step()

            # Train generator
            sr = model(lr)
            hr_c, sr_c = match_size(hr, sr)
            d_fake = disc(sr_c)
            g_loss = srgan_generator_loss(d_fake, sr_c, hr_c, adv_weight=args.adv_weight)
            optim_g.zero_grad()
            g_loss.backward()
            optim_g.step()

            loss_val = g_loss
            metric_psnr = psnr(sr_c.detach(), hr_c.detach()).item()
        else:
            use_amp = args.model == "swinir" and args.amp and scaler is not None
            optim_g.zero_grad()
            if use_amp:
                with autocast():
                    sr = model(lr)
                    hr_c, sr_c = match_size(hr, sr)
                    loss_val = pixel_mse_loss(sr_c, hr_c)
                scaler.scale(loss_val).backward()
                scaler.step(optim_g)
                scaler.update()
            else:
                sr = model(lr)
                hr_c, sr_c = match_size(hr, sr)
                loss_val = pixel_mse_loss(sr_c, hr_c)
                loss_val.backward()
                optim_g.step()
            metric_psnr = psnr(sr_c.detach(), hr_c.detach()).item()

        loss_meter.update(loss_val.item(), n=lr.size(0))
        psnr_meter.update(metric_psnr, n=lr.size(0))
        mse_meter.update(pixel_mse_loss(sr_c.detach(), hr_c.detach()).item(), n=lr.size(0))
        ssim_meter.update(ssim(sr_c.detach(), hr_c.detach()).item(), n=lr.size(0))

    return loss_meter.avg, psnr_meter.avg, mse_meter.avg, ssim_meter.avg


def eval_one_epoch(args, model, disc, loader, device):
    model.eval()
    if disc is not None:
        disc.eval()

    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    mse_meter = AverageMeter()
    ssim_meter = AverageMeter()

    with torch.no_grad():
        for batch in tqdm(loader, desc="val", leave=False, disable=not args.is_main):
            if batch is None:
                continue
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            lr, hr = prepare_batch(lr, hr, args.model, args.use_y_channel)
            if args.model == "swinir":
                val_ps = args.val_patch_size if args.val_patch_size > 0 else args.patch_size
                if val_ps > 0:
                    lr, hr = center_crop_pair(lr, hr, args.scale, val_ps)

            if args.model == "srgan":
                sr = model(lr)
                hr_c, sr_c = match_size(hr, sr)
                d_fake = disc(sr_c)
                loss_val = srgan_generator_loss(d_fake, sr_c, hr_c, adv_weight=args.adv_weight)
            else:
                use_amp = args.model == "swinir" and args.amp
                if use_amp:
                    with autocast():
                        sr = model(lr)
                else:
                    sr = model(lr)
                hr_c, sr_c = match_size(hr, sr)
                loss_val = pixel_mse_loss(sr_c, hr_c)

            metric_psnr = psnr(sr_c, hr_c).item()
            loss_meter.update(loss_val.item(), n=lr.size(0))
            psnr_meter.update(metric_psnr, n=lr.size(0))
            mse_meter.update(pixel_mse_loss(sr_c, hr_c).item(), n=lr.size(0))
            ssim_meter.update(ssim(sr_c, hr_c).item(), n=lr.size(0))

    return loss_meter.avg, psnr_meter.avg, mse_meter.avg, ssim_meter.avg


def save_plots(history, out_dir):
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_psnr = [h["train_psnr"] for h in history]
    val_psnr = [h["val_psnr"] for h in history]
    train_mse = [h["mse_train"] for h in history] if "mse_train" in history[0] else None
    val_mse = [h["mse_val"] for h in history] if "mse_val" in history[0] else None
    train_ssim = [h["ssim_train"] for h in history] if "ssim_train" in history[0] else None
    val_ssim = [h["ssim_val"] for h in history] if "ssim_val" in history[0] else None

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"))
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_psnr, label="train_psnr")
    plt.plot(epochs, val_psnr, label="val_psnr")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("psnr")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "psnr_curve.png"))
    plt.close()

    if train_mse is not None and val_mse is not None:
        plt.figure(figsize=(8, 4))
        plt.plot(epochs, train_mse, label="train_mse")
        plt.plot(epochs, val_mse, label="val_mse")
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("mse")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "mse_curve.png"))
        plt.close()

    if train_ssim is not None and val_ssim is not None:
        plt.figure(figsize=(8, 4))
        plt.plot(epochs, train_ssim, label="train_ssim")
        plt.plot(epochs, val_ssim, label="val_ssim")
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("ssim")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "ssim_curve.png"))
        plt.close()


def setup_ddp(rank, world_size):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def main_worker(rank, world_size, args):
    args.rank = rank
    args.world_size = world_size
    args.is_main = rank == 0
    set_seed(args.seed + rank)

    if world_size > 1:
        setup_ddp(rank, world_size)
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    else:
        device = torch.device(args.device)

    out_dir = Path(args.out_dir)
    weights_dir = out_dir / "weights"
    samples_dir = out_dir / "samples"
    if args.is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        weights_dir.mkdir(parents=True, exist_ok=True)
        samples_dir.mkdir(parents=True, exist_ok=True)

    dataset = SuperResDataset(args.lr_root, args.hr_root)
    splits = split_indices(len(dataset), ratios=args.split, seed=args.seed, shuffle=True)
    train_idx, val_idx = splits[0], splits[1]

    train_sampler = None
    val_sampler = None
    if world_size > 1:
        train_sampler = DistributedSampler(Subset(dataset, train_idx), num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(Subset(dataset, val_idx), num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        Subset(dataset, train_idx) if train_sampler is None else train_sampler.dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_same_size,
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx) if val_sampler is None else val_sampler.dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_same_size,
        sampler=val_sampler,
    )

    # Determine channels
    in_channels = 1 if args.use_y_channel else 3
    out_channels = in_channels

    # For SwinIR, pass the LR spatial size as img_size
    img_size = None
    if args.model == "swinir":
        sample_lr = dataset[0]["lr"]
        h, w = sample_lr.shape[1], sample_lr.shape[2]
        ws = args.swinir_window
        # SwinIR requires init img_size divisible by window_size
        h = ((h + ws - 1) // ws) * ws
        w = ((w + ws - 1) // ws) * ws
        img_size = (h, w)

    model, disc = build_models(args, in_channels, out_channels, img_size=img_size)
    model = model.to(device)
    if disc is not None:
        disc = disc.to(device)

    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)
        if disc is not None:
            disc = torch.nn.parallel.DistributedDataParallel(disc, device_ids=[rank], output_device=rank)

    optim_g = torch.optim.Adam(model.parameters(), lr=args.lr)
    optim_d = None
    if disc is not None:
        optim_d = torch.optim.Adam(disc.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=(args.model == "swinir" and args.amp))

    best_psnr = -1.0
    start_epoch = 1
    if args.weights:
        ckpt = torch.load(args.weights, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optim_g.load_state_dict(ckpt["optimizer"])
        if disc is not None and "discriminator" in ckpt:
            disc.load_state_dict(ckpt["discriminator"])
        if disc is not None and optim_d is not None and "optimizer_d" in ckpt:
            optim_d.load_state_dict(ckpt["optimizer_d"])
        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1
        if "val_psnr" in ckpt:
            best_psnr = float(ckpt["val_psnr"])
    history = []

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_loss, train_psnr, train_mse, train_ssim = train_one_epoch(
                args, model, disc, train_loader, optim_g, optim_d, device, scaler=scaler
            )

            # Disable checkpointing during eval to avoid extra overhead/warnings
            if args.model == "swinir" and args.swinir_checkpoint:
                target = model.module if hasattr(model, "module") else model
                for layer in target.layers:
                    layer.residual_group.use_checkpoint = False

            val_loss, val_psnr, val_mse, val_ssim = eval_one_epoch(args, model, disc, val_loader, device)

            if args.model == "swinir" and args.swinir_checkpoint:
                target = model.module if hasattr(model, "module") else model
                for layer in target.layers:
                    layer.residual_group.use_checkpoint = True

            if args.is_main:
                row = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_psnr": train_psnr,
                    "val_psnr": val_psnr,
                    "mse_train": train_mse,
                    "mse_val": val_mse,
                    "ssim_train": train_ssim,
                    "ssim_val": val_ssim,
                }
                history.append(row)
                write_csv_row(str(out_dir / "metrics.csv"), row)

                # Save sample images
                if args.save_samples_every > 0 and epoch % args.save_samples_every == 0:
                    sample = next(iter(val_loader))
                    lr = sample["lr"].to(device)
                    hr = sample["hr"].to(device)
                    lr, hr = prepare_batch(lr, hr, args.model, args.use_y_channel)
                    if args.model == "swinir":
                        val_ps = args.val_patch_size if args.val_patch_size > 0 else args.patch_size
                        if val_ps > 0:
                            lr, hr = center_crop_pair(lr, hr, args.scale, val_ps)
                    with torch.no_grad():
                        if args.model == "swinir" and args.amp:
                            with autocast():
                                sr = model(lr)
                        else:
                            sr = model(lr)
                    hr_c, sr_c = match_size(hr, sr)
                    lr_c, _ = match_size(lr, sr_c)
                    save_image_triplet(lr_c, sr_c, hr_c, str(samples_dir / f"epoch_{epoch:03d}.png"))

                # Save checkpoints: best and last only
                state = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optim_g.state_dict(),
                    "val_psnr": val_psnr,
                    "args": vars(args),
                }
                if disc is not None:
                    state["discriminator"] = disc.state_dict()
                    state["optimizer_d"] = optim_d.state_dict()

                save_checkpoint(state, str(weights_dir / "last.pt"))

                if val_psnr > best_psnr:
                    best_psnr = val_psnr
                    save_checkpoint(state, str(weights_dir / "best.pt"))

                save_plots(history, str(out_dir))
                print(
                    f"Epoch {epoch}: "
                    f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                    f"train_psnr={train_psnr:.2f} val_psnr={val_psnr:.2f} "
                    f"train_mse={train_mse:.6f} val_mse={val_mse:.6f} "
                    f"train_ssim={train_ssim:.4f} val_ssim={val_ssim:.4f}"
                )
    finally:
        cleanup_ddp()


def main():
    args = parse_args()
    args.rank = 0
    args.world_size = 1
    args.is_main = True
    if args.gpus > 1:
        mp.spawn(main_worker, nprocs=args.gpus, args=(args.gpus, args))
    else:
        main_worker(0, 1, args)


if __name__ == "__main__":
    main()
