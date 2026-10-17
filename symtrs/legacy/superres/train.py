"""The super-resolution trainer: models, data loading, losses, metrics and the loop.

Predates the `symtrs` package and now lives inside it (see `symtrs/legacy/README.md`),
launched as `python -m symtrs.legacy.superres.train` by `symtrs.model.superres`. The code
below is `train_superresolution.py` from the repository root with the changes listed in
`symtrs/legacy/README.md` ("Changes after the move"): the header and imports, then crops for
every model, whole-image (optionally windowed) validation and test, and an optional held-out
validation split. Read that list before assuming a number came from the original loop.
"""

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

# `ROOT` is only the default for `--out_dir` now; `symtrs` always passes `--out_dir`,
# so it is the hand-run default and nothing else.
ROOT = Path(__file__).resolve().parent

from symtrs.engine.tiling import predict_tiled
from symtrs.legacy.dataloader import SuperResDataset, split_indices
from symtrs.legacy.superres.models import (
    SRCNN,
    AutoencoderSR,
    SRGANGenerator,
    SRGANDiscriminator,
    SwinIR,
)
from symtrs.legacy.superres.utils import (
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
    # Held-out evaluation. Without these, the only number this script produces comes from
    # the validation split it carves out of `--lr_root` itself, so a checkpoint is chosen
    # on data it trained near and nothing is ever scored on the SyMTRS test split. The
    # `symtrs` front-end sets both to <root>/test, and reads `test_metrics.json` back.
    p.add_argument("--test_lr_root", type=str, default="", help="LR images of the held-out test split")
    p.add_argument("--test_hr_root", type=str, default="", help="HR images of the held-out test split")
    p.add_argument("--eval_only", action="store_true",
                   help="skip training and score out_dir/weights/best.pt on the test split")
    p.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use (1 for single-GPU, >1 for DDP)")
    # Validation set. Without these the script carves `--split` out of `--lr_root`, i.e. out of
    # train/, which AGENTS.md ground rule 6 forbids for the SyMTRS layout; the `symtrs`
    # front-end passes <root>/val here so the split lives in the dataset.
    p.add_argument("--val_lr_root", type=str, default="", help="LR images of the validation split")
    p.add_argument("--val_hr_root", type=str, default="", help="HR images of the validation split")
    p.add_argument("--patch_size", type=int, default=0,
                   help="LR patch size for TRAINING, every model (0 = whole frames; 2048 px frames "
                        "do not fit a 12 GB card). The HR patch is patch_size * scale.")
    p.add_argument("--val_patch_size", type=int, default=0,
                   help="LR size of a deterministic CENTRE crop to validate/test on, for speed only "
                        "(0 = the whole image, never the training patch: ground rule 11)")
    p.add_argument("--eval_tile_size", type=int, default=0,
                   help="validate/test the whole image in overlapping windows of this many LR pixels "
                        "(0 = one forward pass). Metrics are computed on the stitched image.")
    p.add_argument("--eval_tile_overlap", type=int, default=0,
                   help="LR pixels shared by neighbouring evaluation windows")
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


def swinir_img_size(args, sample_lr):
    """The `img_size` SwinIR is built with: the training patch, else the full LR frame.

    SwinIR precomputes its shifted-window attention masks for `img_size` and stores them in
    the state dict as buffers, one `[windows, 49, 49]` tensor per shifted block. Built for a
    whole 2048 px frame that is ~10 GB (the x1 smoke OOM'd just constructing the model) and
    it ends up in every checkpoint. Any other input size recomputes the mask on the fly, so
    the size only has to be the one the training step uses: the crop. Training, resume and
    `evaluate_on_test` all call this, so the buffers always match the checkpoint.
    """
    ws = args.swinir_window
    if args.patch_size > 0:
        h = w = args.patch_size
    else:
        h, w = sample_lr.shape[1], sample_lr.shape[2]
    return (((h + ws - 1) // ws) * ws, ((w + ws - 1) // ws) * ws)


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


def prepare_input(lr, model_name, use_y, out_size):
    """The tensor the network sees for an LR image whose target is `out_size` (H, W).

    SRCNN and the autoencoder refine a bicubic upsample, so they receive `out_size`; SRGAN and
    SwinIR upsample themselves and receive the LR image. One function for training and for
    evaluation, so a windowed evaluation cannot drift from what the loop trained on.
    """
    if use_y:
        lr = rgb_to_y(lr)
    if model_name in {"srcnn", "autoencoder"}:
        lr = F.interpolate(lr, size=tuple(out_size), mode="bicubic", align_corners=False)
    return lr


def prepare_batch(lr, hr, model_name, use_y):
    if use_y:
        hr = rgb_to_y(hr)
    return prepare_input(lr, model_name, use_y, hr.shape[-2:]), hr


#: SRCNN has no padding in its 9x9 and 5x5 convolutions, so it returns 12 px less than it is
#: given (6 per side). Training scores the interior (`match_size` centre-crops the target); an
#: evaluation that stitches windows needs an answer exactly as large as the window, so the
#: window is reflect-padded by what the valid convolutions will eat.
SRCNN_MARGIN = 6


def super_resolve(args, model, lr):
    """Upscale one LR image `[1, 3, h, w]` to `[1, C, h*scale, w*scale]`, whole or in windows.

    The single inference path of validation, test and the sample panels. The whole image goes
    through in one pass unless `--eval_tile_size` asks for overlapping windows of that many LR
    pixels, which `symtrs.engine.tiling.predict_tiled` blends with a Hann weight and stitches
    at `scale` times the input size. Metrics are then computed on the stitched full image.
    """
    scale = args.scale

    def forward(tile):
        x = prepare_input(tile, args.model, args.use_y_channel, (tile.shape[-2] * scale, tile.shape[-1] * scale))
        if args.model == "srcnn":
            x = F.pad(x, (SRCNN_MARGIN,) * 4, mode="reflect")
        if args.model == "swinir" and args.amp:
            with autocast():
                sr = model(x)
        else:
            sr = model(x)
        return sr.float()

    h, w = lr.shape[-2:]
    if args.eval_tile_size > 0 and (h > args.eval_tile_size or w > args.eval_tile_size):
        return predict_tiled(
            lr, forward, args.eval_tile_size, overlap=args.eval_tile_overlap, scale=scale
        )
    return forward(lr)


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
    ps_hr = patch_size * scale
    if h_lr < patch_size or w_lr < patch_size or h_hr < ps_hr or w_hr < ps_hr:
        # Used to return the whole frame, which turned a too-large patch into a silent
        # whole-image step (and an out-of-memory error far from its cause at 2048 px).
        raise ValueError(
            f"--patch_size {patch_size} (x{scale} = {ps_hr} target px) does not fit a "
            f"{h_lr}x{w_lr} LR / {h_hr}x{w_hr} HR frame; lower train.patch_size"
        )

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
        # Crop first, for EVERY model, and on the CPU: this used to happen only for SwinIR and
        # after the bicubic upsample, so SRCNN / the autoencoder / SRGAN trained on whole
        # 2048 px frames (31.75 GiB for one SRCNN activation at batch 16). Cropping the LR
        # image before `prepare_batch` also keeps the upsample patch-sized.
        lr, hr = random_crop_pair(batch["lr"], batch["hr"], args.scale, args.patch_size)
        lr = lr.to(device)
        hr = hr.to(device)
        lr, hr = prepare_batch(lr, hr, args.model, args.use_y_channel)

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
    """Score `loader` on WHOLE images, one at a time, by the same path for val and test.

    This used to centre-crop to `--val_patch_size` (or, at 0, to the TRAINING patch) for
    SwinIR and push whole batches of whole frames through the other models, and it computed
    PSNR from the mean MSE of a batch. Now every image goes through `super_resolve` alone
    (whole, or stitched from windows when `--eval_tile_size` is set), is scored at full size,
    and PSNR / SSIM / MSE are per-image values averaged over images, which is what a test
    table means by them. `--val_patch_size > 0` is the one remaining way to score a centre
    crop, for a quick run.
    """
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
            for i in range(batch["lr"].size(0)):
                lr = batch["lr"][i : i + 1].to(device)
                hr = batch["hr"][i : i + 1].to(device)
                if args.val_patch_size > 0:
                    lr, hr = center_crop_pair(lr, hr, args.scale, args.val_patch_size)
                if args.use_y_channel:
                    hr = rgb_to_y(hr)
                sr = super_resolve(args, model, lr)
                hr_c, sr_c = match_size(hr, sr)
                if args.model == "srgan":
                    d_fake = disc(sr_c)
                    loss_val = srgan_generator_loss(d_fake, sr_c, hr_c, adv_weight=args.adv_weight)
                else:
                    loss_val = pixel_mse_loss(sr_c, hr_c)

                loss_meter.update(loss_val.item())
                # Scored on the image a user would save: the network output is unbounded, the
                # written PNG is clamped to [0, 1], and PSNR is defined for data range 1.
                sr_m = sr_c.clamp(0.0, 1.0)
                psnr_meter.update(psnr(sr_m, hr_c).item())
                mse_meter.update(pixel_mse_loss(sr_m, hr_c).item())
                ssim_meter.update(ssim(sr_m, hr_c).item())

    return loss_meter.avg, psnr_meter.avg, mse_meter.avg, ssim_meter.avg


def evaluate_on_test(args, device):
    """Score `weights/best.pt` on the held-out test split.

    Called only when `--test_lr_root` and `--test_hr_root` are both given. The model is
    rebuilt unwrapped rather than reused: under `--gpus > 1` the training model is a
    `DistributedDataParallel`, and scoring a DDP wrapper on a single process would need a
    gather for no benefit. `build_models` is deterministic, so this rebuilds exactly the
    architecture the checkpoint was written from.

    Writes `out_dir/test_metrics.json` and returns its contents. The `symtrs` front-end
    turns that into the `summary.json` the rest of the package reads, so a
    super-resolution row in the results table carries the same kind of recorded test
    number as a depth or segmentation row.
    """
    import json

    best_path = Path(args.out_dir) / "weights" / "best.pt"
    if not best_path.is_file():
        print(f"[test] no checkpoint at {best_path}; skipping the test evaluation")
        return {}

    test_dataset = SuperResDataset(args.test_lr_root, args.test_hr_root)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_same_size,
    )

    in_channels = 1 if args.use_y_channel else 3
    img_size = None
    if args.model == "swinir":
        img_size = swinir_img_size(args, test_dataset[0]["lr"])

    model, disc = build_models(args, in_channels, in_channels, img_size=img_size)
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    if disc is not None:
        disc = disc.to(device)
        if "discriminator" in checkpoint:
            disc.load_state_dict(checkpoint["discriminator"])

    loss, test_psnr, test_mse, test_ssim = eval_one_epoch(args, model, disc, test_loader, device)

    metrics = {
        "test_loss": loss,
        "psnr": test_psnr,
        "mse": test_mse,
        "ssim": test_ssim,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "images": len(test_dataset),
    }
    out_path = Path(args.out_dir) / "test_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(
        f"[test] {len(test_dataset)} images: psnr={test_psnr:.4f} ssim={test_ssim:.4f} "
        f"mse={test_mse:.6f} -> {out_path}"
    )
    return metrics


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
    if args.val_lr_root and args.val_hr_root:
        # The split lives in the dataset (ground rule 6): train on all of train/, validate on
        # val/. Two datasets, so the index lists are into different datasets.
        val_dataset = SuperResDataset(args.val_lr_root, args.val_hr_root)
        train_idx = list(range(len(dataset)))
        val_idx = list(range(len(val_dataset)))
    else:
        print(f"[split] no --val_lr_root/--val_hr_root: carving {args.split} out of {args.lr_root} "
              "(the original behaviour; the symtrs front-end passes val/ whenever it exists)")
        val_dataset = dataset
        splits = split_indices(len(dataset), ratios=args.split, seed=args.seed, shuffle=True)
        train_idx, val_idx = splits[0], splits[1]

    train_sampler = None
    val_sampler = None
    if world_size > 1:
        train_sampler = DistributedSampler(Subset(dataset, train_idx), num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(Subset(val_dataset, val_idx), num_replicas=world_size, rank=rank, shuffle=False)

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
        Subset(val_dataset, val_idx) if val_sampler is None else val_sampler.dataset,
        batch_size=1,
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
        img_size = swinir_img_size(args, dataset[0]["lr"])

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
        # The best score so far, not the last epoch's: resuming from `last.pt` with that epoch's
        # val_psnr let a worse later epoch overwrite best.pt. Older checkpoints lack the key.
        if "best_psnr" in ckpt:
            best_psnr = float(ckpt["best_psnr"])
        elif "val_psnr" in ckpt:
            best_psnr = float(ckpt["val_psnr"])
    history = []

    try:
        # `--eval_only` empties the loop rather than guarding it, so the test evaluation
        # below still runs and there is exactly one code path to it.
        for epoch in (() if args.eval_only else range(start_epoch, args.epochs + 1)):
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
                    lr_raw = sample["lr"][:1].to(device)
                    hr = sample["hr"][:1].to(device)
                    if args.use_y_channel:
                        hr = rgb_to_y(hr)
                    with torch.no_grad():
                        sr = super_resolve(args, model, lr_raw)
                    hr_c, sr_c = match_size(hr, sr)
                    # A whole 2048 px triplet is ~25 MB of PNG per epoch; the panel is a
                    # visual check, so show the centre 512 px of the whole-image result.
                    side = min(512, sr_c.shape[-2], sr_c.shape[-1])
                    sr_c, hr_c = center_crop_pair(sr_c, hr_c, 1, side)
                    lr_view = rgb_to_y(lr_raw) if args.use_y_channel else lr_raw
                    lr_c, _ = center_crop_pair(lr_view, hr_c, args.scale, max(1, side // args.scale))
                    lr_c = F.interpolate(lr_c, size=sr_c.shape[-2:], mode="nearest")
                    save_image_triplet(lr_c, sr_c, hr_c, str(samples_dir / f"epoch_{epoch:03d}.png"))

                # Save checkpoints: best and last only
                state = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optim_g.state_dict(),
                    "val_psnr": val_psnr,
                    "best_psnr": max(best_psnr, val_psnr),
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

    # After the loop, not inside it: the held-out split is scored once, from the
    # checkpoint that validation selected, and only on the process that owns the run.
    if args.is_main and args.test_lr_root and args.test_hr_root:
        evaluate_on_test(args, device)


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
