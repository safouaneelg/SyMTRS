# Moving the legacy super-resolution code inside the `symtrs` package

Date: 2026-10-15
Status: proposed, awaiting author review

## Purpose

The super-resolution task is the one SyMTRS task whose implementation sits *beside* the
package rather than inside it. Its four networks live in `models/superresolution/`, its
dataset in `dataloader/`, its losses and metrics in `utils/superresolution/`, and its
training loop in `train_superresolution.py` at the repository root. `symtrs` reaches all
of it through a subprocess, and `train_superresolution.py` reaches it by putting the
repository root on `sys.path`.

That arrangement has three costs, in increasing order of how much they matter:

1. **It does not ship.** `[tool.setuptools.packages.find]` collects `symtrs*` only, so an
   installed `symtrs` has a super-resolution task that cannot run: the script and the three
   trees are not in the distribution. `pip install -e .` hides this, which is exactly the
   kind of defect that survives to release.
2. **The task depends on a top-level name.** `dataloader`, `models` and `utils` are the
   most collision-prone names a Python project can put at its root; `utils` is *why*
   `train_superresolution.py` is launched as a subprocess and never imported. That
   constraint is load-bearing for the front-end's design and would disappear if the names
   did.
3. **It reads as third-party code.** `models/`, `utils/` and `dataloader/` are this
   repository's own code (commits 08c9931, a2318d5, 61c644a), but their location makes
   them look like another vendored tree next to `pytorch-CycleGAN-and-pix2pix/`. They are
   not, and the difference matters: the vendored tree is byte-identical to upstream on
   purpose and must stay that way, while this code is the authors' own and belongs in the
   package.

This document specifies moving that code under `symtrs.legacy.*`, keeping the vendored
CycleGAN/pix2pix tree exactly where it is.

**Success is a number that does not move.** The change is a relocation, not a
refactor: after it, the same config on the same data must produce the same
`test_metrics.json` as before it, and the `symtrs` command line, config schema, run
artefacts and SDK must be indistinguishable to a caller.

## The rule this change follows: moved, never ported

Every file is relocated **byte-identical**. Nothing is rewritten, reformatted, retyped,
renamed, or given type hints or docstrings on the way. Provenance is the point: a reader
in three years has to be able to check that the network that produced a checkpoint is the
network in the paper, and a move keeps that checkable while a reorganisation does not.

This is possible because every import inside the three trees is already relative
(`from .srcnn import SRCNN`, `from .utils import read_image`, `from .losses import ...`),
so no module needs editing for its new home. Only the trainer's four *top-level* imports
change, and it is the one file whose text changes at all.

## Target layout

```text
symtrs/
  legacy/                       # code that predates this package; moved, not ported
    __init__.py                 # why this tree exists and the one rule: never import it
                                #   from anywhere but itself
    README.md                   # provenance table + how to verify the move from git
    superres/
      __init__.py               # points at train.py; states the byte-identity rule
      train.py                  # <- train_superresolution.py (imports + sys.path only)
      models/                   # <- models/superresolution/       (byte-identical)
        __init__.py  srcnn.py  autoencoder_sr.py  srgan.py  swinir.py
      utils/                    # <- utils/superresolution/        (byte-identical)
        __init__.py  losses.py  metrics.py  image_utils.py  train_utils.py
    dataloader/                 # <- dataloader/                   (byte-identical)
      __init__.py  utils.py  split.py  superres_dataset.py
      depth_dataset.py  domain_adapt_dataset.py  README.md
```

The two directory names `models/` and `utils/` are kept **inside** the legacy subtree on
purpose: they are the originals, so the provenance mapping is a pure move, and inside
`symtrs/legacy/superres/` they cannot collide with anything (they are only ever imported
as `symtrs.legacy.superres.models` / `...utils`). Renaming them to something prettier
(`nets/`, `measures/`) would buy nothing and cost the mechanical mapping.

`symtrs/legacy/__init__.py` and `symtrs/legacy/superres/__init__.py` are new files that
import nothing. That is deliberate: `import symtrs.legacy` must not drag torch, matplotlib
or the four networks into a process that only wanted to know the tree is there — the
imports that cost something belong in `train.py`, which is only ever started as its own
process. `symtrs/legacy/superres/models/__init__.py` is the moved one and is the entry
point for the networks.

At the repository root afterwards:

| path | state |
| --- | --- |
| `train_superresolution.py` | **shim** — imports `main` from the new location, no logic |
| `dataloader/__init__.py` | **shim** — re-exports, so the documented `from dataloader import ...` still resolves |
| `models/` | **removed** — its single consumer moved, and no documented import path names it |
| `utils/` | untouched in this change (only `__init__.py` and the unreferenced `genai/` remain) — see Open question |
| `pytorch-CycleGAN-and-pix2pix/` | untouched, byte-identical, still the domain-adaptation implementation |

## Provenance table

| from | to | edit |
| --- | --- | --- |
| `models/superresolution/__init__.py` | `symtrs/legacy/superres/models/__init__.py` | none |
| `models/superresolution/{srcnn,autoencoder_sr,srgan,swinir}.py` | `symtrs/legacy/superres/models/` | none |
| `utils/superresolution/__init__.py` | `symtrs/legacy/superres/utils/__init__.py` | none |
| `utils/superresolution/{losses,metrics,image_utils,train_utils}.py` | `symtrs/legacy/superres/utils/` | none |
| `dataloader/*.py`, `dataloader/README.md` | `symtrs/legacy/dataloader/` | none |
| `train_superresolution.py` | `symtrs/legacy/superres/train.py` | module header only: docstring, import block, the two `sys.path` lines removed, stale first line and its BOM dropped |

The files keep their UTF-8 BOMs, because keeping them is the byte-identity rule and Python
accepts a BOM in a source file. Anything that *reads* those files as text (a test, a
script) must therefore open them with `encoding="utf-8-sig"`.

`train.py`'s new import block replaces three top-level imports:

```python
from symtrs.legacy.dataloader import SuperResDataset, split_indices
from symtrs.legacy.superres.models import (
    SRCNN, AutoencoderSR, SRGANGenerator, SRGANDiscriminator, SwinIR,
)
from symtrs.legacy.superres.utils import (
    AverageMeter, match_size, pixel_mse_loss, psnr, ssim, rgb_to_y,
    save_checkpoint, save_image_triplet, set_seed,
    srgan_discriminator_loss, srgan_generator_loss, write_csv_row,
)
```

and deletes the two `sys.path` lines:

```python
sys.path.append(str(ROOT))
sys.path.append(r"E:\\dataloader")
```

`ROOT = Path(__file__).resolve().parent` **stays**: line 56 uses it as `--out_dir`'s default.
That default is left exactly as it is — changing it would be a behaviour change — which is
why `ROOT` survives the move even though the line that sat beside it does not.

The Windows path is not a typo in this document; it is what the file contains today, along
with a `# E:\train_superresolution.py` first line. Both go, and with them the file's BOM.

`train.py` needs no structural change to become a module: it already ends with
`def main():` and `if __name__ == "__main__": main()`, and `parse_args()` reads `sys.argv`,
so `python -m symtrs.legacy.superres.train` runs it as it stands. Nothing inside any
function changes — the edits are the module header and nothing else.

## The launch

`SuperResolution.train_command` now builds:

```text
<sys.executable> -m symtrs.legacy.superres.train --lr_root ... --hr_root ... ...
```

instead of pointing at a file in the repository root. This is what makes a non-editable
install work, and it removes the front-end's only dependency on the checkout's layout.
`symtrs.legacy.superres.train` is a name inside the package, so if `symtrs` is importable
the trainer is present — the old `SuperResolutionError("the super-resolution script is
missing from ...")` guard was a check for a file that lived outside the package and is
replaced by a `find_spec` preflight on the module name (the same legibility trick
`upstream.missing_imports()` uses for `dominate`/`wandb`), so a stripped or broken install
still fails with an instruction rather than a `ModuleNotFoundError` from a child process.

**It stays a subprocess.** The move makes an in-process call possible for the first time,
and it is still the wrong choice: the legacy loop owns its own RNG (`set_seed`), its own
matplotlib backend (`Agg`), its own `torch.multiprocessing` start-up and its own
`argparse`, all of which would become the CLI process's state, and a crash in it would
take the CLI down with it. The docstring's current *reason* ("the script puts the
repository root on `sys.path`") is no longer true and must be rewritten to these reasons
rather than left as a stale justification.

DDP (`--gpus N`) uses `torch.multiprocessing.spawn` inside the child. Spawned ranks
unpickle the worker by module path, and `multiprocessing.spawn` passes the parent's
`sys.path` to each rank, so `-m` discovery carries over unchanged. This cannot be verified
on this machine (no CUDA, and upstream's path is NCCL-only); it is verified by the argv
and by an import check, and belongs on the first-GPU-run checklist.

## What callers see

Unchanged: `symtrs train/eval/sweep/tables --task superresolution`, every config key,
`summary.json`, `test_metrics.json`, the run-directory artefacts, `symtrs doctor`,
the SDK (`autoloader("srcnn").train(...)`), and the argv after the launch prefix.

Changed, and worth saying out loud:

- `python train_superresolution.py --lr_root ...` still works — it is the shim — but the
  same command also exists as `python -m symtrs.legacy.superres.train`.
- `from dataloader import SuperResDataset` still works through the root shim; the code now
  lives at `symtrs.legacy.dataloader`.
- `from models.superresolution import SRCNN` **stops working**: `models/` is removed, the
  new path is `symtrs.legacy.superres.models`. Nothing in this repository, in the configs,
  in the tests or in the README uses that import, so the only exposure is an out-of-repo
  script of the author's. Said here so it is a decision and not a surprise.
- `symtrs` installed as a wheel now contains the super-resolution task.

## Non-goals

1. **The vendored tree is not touched.** `pytorch-CycleGAN-and-pix2pix/` stays
   byte-identical at `2a7afba`, at its current path. Its byte-identity is the claim behind
   every domain-adaptation number ("we ran upstream's code"); moving or porting it would
   destroy the claim, and `symtrs setup-adapt --check` verifies it.
2. **No code is rewritten, tidied or made consistent.** Not one model, loss, metric,
   dataset or loop line, in the moved trees or at the root.
3. **No behaviour change.** No new flag, no new config key, no new metric, no changed
   default. If a step appears to need one, it is a separate change with its own spec.
4. **No dead code is deleted.** See below.
5. **`symtrs/model/superres/` keeps its job** — folder mapping, launch, `summary.json` —
   and gains no responsibility for the models.

## Open question: the dead code (needs a decision, not part of the move)

Moving `dataloader/` wholesale brings two unused modules along, and `utils/` keeps a third
piece:

| path | lines | who uses it |
| --- | --- | --- |
| `utils/genai/{__init__,image_utils,losses,metrics,train_utils}.py` | 169 | nothing in this repository |
| `symtrs/legacy/dataloader/depth_dataset.py` | 61 | only `dataloader/__init__.py`'s re-export |
| `symtrs/legacy/dataloader/domain_adapt_dataset.py` | 62 | only `dataloader/__init__.py`'s re-export |

Deleting them is a yes/no that this spec deliberately does not take, because the author may
use them from scripts outside this repository, and because "unused here" is not the same as
"unused". Recommendation: **delete `utils/genai/` and the two datasets in a separate,
clearly-labelled commit after the move**, keeping the root `dataloader/__init__.py` shim
pointing at only what remains — or, if any of them is still wanted, keep all three and
accept that `symtrs/legacy/` is where the author's other experiments live. The move itself
is the same either way, so this does not block the plan.

Deleting `utils/genai/` means deleting `utils/` entirely, including its empty
`__init__.py`. That is the desirable end state: after it, this repository has no
top-level `utils` at all, and the reason `train_superresolution.py` used to be a
subprocess — a top-level package name the library must not inherit — no longer exists in
this checkout. Until then the `utils/` package stays, unreferenced, exactly as it is now.

## Plumbing sites

Every place that names the old location, and what happens to it:

| file | what changes |
| --- | --- |
| `symtrs/model/superres/runner.py` | module docstring (lines 1-31); `repo_root()` and `script_path()` replaced by a `TRAIN_MODULE` constant; `train_command`'s existence guard and launcher; `run_super_resolution`'s subprocess-justification paragraph; the `"train_superresolution.py exited with ..."` message; `__all__` |
| `symtrs/model/superres/__init__.py` | module docstring; drop `repo_root`/`script_path` from the imports and `__all__` |
| `symtrs/commands/_common.py` | `print_superresolution_command` (282-303): the printed label and the `train_command(...)[2:]` slice, which no longer skips what it thinks it skips |
| `symtrs/engine/trainer.py` | the 42-49 comment: which file owns the loop, and why it is still driven as a subprocess |
| `symtrs/registry.py` | the 280-281 comment above the four SR specs |
| `configs/generate_grid.py` | the super-resolution header text (~1054), which says the script "at the repository root owns the model set"; regenerate the 16 SR YAMLs so `--check` stays green |
| `tests/test_superres.py` | `command[1]` is no longer a script path; the registry/`choices=` drift test reads the trainer from its new path (open with `utf-8-sig`) |
| `AGENTS.md` | §1 bullet, §5 tree, §7's `sys.path` trap (rewritten as history: the trap is gone, the lesson is not), §3's `compileall` line if it needs the new path |
| `README.md` | the tree listing (~35), the SDK comment (~455), the direct-command block (~602, now the shim) and the legacy-loader imports (~636) |
| `docs/superpowers/specs/2026-10-15-symtrs-tasks-design.md` | a one-line "superseded for the super-resolution layout by <this document>" note; the body is history and is not rewritten |
| `pyproject.toml` | **no change** — `include = ["symtrs*"]` already covers `symtrs.legacy.*`, which is the point of the move |

## Testing

The suite must stay green and grow. New tests, all torch-free, in
`tests/test_legacy_layout.py`:

- **Hygiene, by source inspection.** No module under `symtrs/legacy/` imports a top-level
  `models`, `utils` or `dataloader`; no module anywhere in `symtrs/` writes to `sys.path`.
  Both fail today, both are the point of the change, and both are asserted as reasons
  rather than values: the comment says the tree must be importable without the repository
  root.
- **The tree is complete and in place.** All four nets (`srcnn`, `autoencoder_sr`, `srgan`,
  `swinir`), all four `utils/superresolution` modules (`losses`, `metrics`, `image_utils`,
  `train_utils`), the trainer, and the loading the trainer uses (`superres_dataset.py`,
  `split.py`, `utils.py`) exist at their new paths, and every moved module imports nothing
  but stdlib, third-party or relative names.
- **The shims are shims.** `train_superresolution.py` contains no `argparse`, no
  `sys.path.append` and no model class; it imports `main` from the new path. A shim that
  grows logic is the drift this design exists to prevent, so the test asserts its absence
  structurally.
- **The front-end launches the module.** `train_command(...)[:3] == [sys.executable, "-m",
  "symtrs.legacy.superres.train"]`, with no `.py` token in the command at all.
- **The `find_spec` preflight** names the module and the fix when the trainer is not
  importable (monkeypatched, the way `tests/test_domain_commands.py` tests
  `missing_imports`).

Updated in `tests/test_superres.py`: the "launched, never imported" pair of assertions
becomes "launched as a module of this package", and the registry-versus-`choices=` test
reads `symtrs/legacy/superres/train.py`.

## Verification

Executable, in order. Steps 1 and 6 are the ones that make this a *move* rather than a
claim of one.

1. **Baseline, before any file moves.** With the dummy dataset:
   `symtrs train configs/superresolution/srcnn_x4.yaml --data-root /tmp/symtrs_dummy
   --device cpu --epochs 1 --batch-size 2 --num-workers 0 --out-dir /tmp/symtrs_move/before`,
   plus the `--dry-run` output. Keep `test_metrics.json`, `summary.json` and the printed
   command.
2. Move the files with `git mv` so the history follows each one.
3. Edit `train.py`'s imports, delete its two `sys.path` lines, add `main()`. Add the
   `symtrs/legacy/__init__.py`, `symtrs/legacy/superres/__init__.py` and `legacy/README.md`
   provenance notes. Remove the emptied `models/`. Add the two root shims.
4. Update the plumbing sites above; regenerate the 16 SR configs.
5. `python -m pytest -q` (green, and longer by the new file), `python -m compileall -q
   symtrs train_superresolution.py`, `python3 configs/generate_grid.py --check`.
6. **The same run again**, same command, `--out-dir /tmp/symtrs_move/after`, and a
   `diff` of `test_metrics.json` against step 1. Expected: byte-identical, because the
   seed, the data, the model and the code are the same. A difference is not to be
   explained away as "close enough" — find it, because this is the only evidence that the
   move is a move.
7. **The other two entry points into the same code**: `symtrs eval
   configs/superresolution/srcnn_x4.yaml --data-root /tmp/symtrs_dummy --eval-split test`
   on the finished run directory (the `--eval_only` path), and `symtrs train ...
   --dry-run` (the command printer). Both exercised against the new launch.
8. **A fresh interpreter that is not the repository root**:
   `cd /tmp && PYTHONPATH=<repo> python -c "import symtrs.legacy.superres.train,
   symtrs.legacy.dataloader"` and `python -m symtrs.legacy.superres.train --help`.
9. **The shims, as a user would type them**: `python train_superresolution.py --help` and
   `python -c "from dataloader import SuperResDataset, read_image"`.
10. `symtrs doctor --data-root /tmp/symtrs_dummy` and `symtrs models` — nothing in the
    registry or the doctor changed, and this proves it.

## Committing, and rollback

Nothing in this repository is committed yet, so the move would otherwise land inside a
diff with the entire `symtrs/` package in it and be unreviewable. Recommendation: commit
the current state first, then make the move its own commit, so `git show --stat` displays
pure renames (`R100` for the byte-identical files, a handful of changed lines for
`train.py`) and `git log --follow` still reaches commit `61c644a`.

Rollback is `git mv` in the other direction plus `git checkout train_superresolution.py`,
which is possible only if the pre-move state was committed — the second reason to commit
before moving.

## Risks

| risk | why it is small | what catches it |
| --- | --- | --- |
| A file is edited "while I am here", and the numbers move | the rule is byte-identity, and the only file that changes is `train.py`, whose diff is its import block | step 6's byte-identical `test_metrics.json`; the hygiene test |
| The child process cannot import the package | `-m` finds it exactly when the `symtrs` CLI does, and the `find_spec` preflight says so when it does not | steps 8 and 9 |
| DDP/spawn breaks under `-m` | spawn transfers `sys.path`; nothing in the worker depends on `__file__`'s directory | not verifiable here — on the first GPU run |
| A caller outside the repository imported `models.superresolution` | no such import exists in this repository; the new path is one line | the provenance note in `README.md` |
| The legacy tree is mistaken for the vendored tree and edited freely | `symtrs/legacy/README.md` and the package docstrings say moved-not-ported, and the vendored tree's byte-identity is asserted elsewhere | the existing vendored-tree tests |
