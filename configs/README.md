# `configs/` - one file per cell of the results grid

Every run in the benchmark is described by a YAML file here. Nothing about a run is
hidden in the code: the file states the task, the model, the resolution, the optimiser and
the evaluation protocol, and `symtrs train <file>` executes exactly that.

```
configs/
  train_sr_x2.yaml            quick-start: the whole path, in one file
  depth/<2048|1024|512|256>/  <model>_<resolution>.yaml   11 models x 4 resolutions
  segmentation/<...>/         <model>_<resolution>.yaml    6 models x 4 resolutions
  domain_adapt/               <model>_<direction>.yaml     2 models x 2 directions
  superresolution/            <model>_x<scale>.yaml        4 models x 4 scales
  generate_grid.py            the tables these 88 files are expanded from
```

```bash
symtrs train configs/depth/2048/midas_dpt_hybrid_2048.yaml --data-root /data/SyMTRS
symtrs train configs/depth/2048/midas_dpt_hybrid_2048.yaml --data-root /data/SyMTRS --dry-run
symtrs sweep --task depth --models midas_dpt_hybrid    # every resolution of one row
symtrs sweep --task depth                              # every cell
```

## The resolution axis

`resolution` names the *pixel size of the data*, not a scale factor, and it means the same
thing in every task:

| `resolution` | folders read | 
| --- | --- |
| 2048 | `rgb_day` + `depth` / `semantic` (native) |
| 1024 | `rgb_day_lr/x2` + `depth_lr/x2` / `semantic_lr/x2` |
| 512 | `rgb_day_lr/x4` + `depth_lr/x4` / `semantic_lr/x4` |
| 256 | `rgb_day_lr/x8` + `depth_lr/x8` / `semantic_lr/x8` |

The four cells of a row are four separate runs over four separate datasets of the same
scenes. They are rows of one table, not stages of a pipeline: nothing in depth or
segmentation upsamples anything. Domain adaptation is native-only, because `rgb_night` has
no `_lr` counterpart.

Super-resolution is the one task where the axis is a factor as well as a size: there
`resolution` names the LR input, and `scale = 2048 / resolution` is derived from it.
`configs/superresolution/srcnn_x4.yaml` therefore carries `resolution: 512`, reads
`rgb_day_lr/x4`, and is scored against native `rgb_day`. `..._x1.yaml` is a ×1 refiner
control row - LR and HR are the same native folder - and not super-resolution.

## Regenerating

`generate_grid.py` holds the model tables (protocol, loss, batch size, memory posture, the
prose) and writes all 88 files. Change a shared hyper-parameter there rather than in 44
placeholders, then:

```bash
python3 configs/generate_grid.py           # rewrite the tree
python3 configs/generate_grid.py --check   # fail if the tree has drifted from the tables
```

The per-model note lives on the native-resolution file of each row; the 1024/512/256 files
carry only what changes, plus the invariant that must *not* change - a relative-depth model
is `align: scale` and `loss: silog` at every resolution, and switching that between two
cells of one row makes them incomparable.
