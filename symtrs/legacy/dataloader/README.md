# Data Loaders (PyTorch)

This folder provides small PyTorch datasets for:
- Monocular depth estimation: `DepthDataset`
- Super‑resolution: `SuperResDataset`
- Day→night domain adaptation: `DomainAdaptDataset`

## Common utilities
- `read_image(path, mode="rgb")` → float tensor [C,H,W] in [0,1]
- `split_indices(n, ratios=(0.8,0.2), seed=42)`
- `split_list(items, ratios=(0.8,0.2), seed=42)`

## Depth (raw or per‑image normalized)
```python
from dataloader import DepthDataset

dset = DepthDataset(
    depth_root=r"SyMTRS/depth",
    hr_root=r"SyMTRS\hr",
    normalize_depth="raw",  # or "per_image_minmax"
)
```

## Super‑resolution
```python
from dataloader import SuperResDataset

dset = SuperResDataset(
    lr_root=r"SyMTRS\lr\x4",
    hr_root=r"SyMTRS\hr",
)
```

## Day→night domain adaptation
```python
from dataloader import DomainAdaptDataset

dset = DomainAdaptDataset(
    day_root=r"SyMTRS\hr",
    night_root=r"SyMTRS\night",
    paired=False,
)
```

## Splitting examples
```python
from dataloader import split_indices

idx_train, idx_val = split_indices(len(dset), ratios=(0.8, 0.2))

idx_train, idx_val, idx_test = split_indices(len(dset), ratios=(0.7, 0.15, 0.15))
```
