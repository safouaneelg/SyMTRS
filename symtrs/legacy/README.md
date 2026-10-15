# `symtrs/legacy/` — code that predates the package, moved inside it unchanged

Everything under this directory is this repository's own code (commits 08c9931, a2318d5,
61c644a), not a third-party tree. It used to sit beside the package at the repository root,
where top-level `utils` and `dataloader` names made it unreachable from an installed
`symtrs` and forced the super-resolution trainer to be launched as a script rather than
imported. It lives inside the package now: **relocated, not ported**, so that the network
behind a published checkpoint stays checkable against the code that trained it.

The one file whose text changed is `superres/train.py`, and only its module header: a
docstring, the import block, and the two `sys.path.append` lines its script form needed. No
model, loss, metric, dataset or training-loop line changed anywhere.

## Where every file came from

| now | before |
| --- | --- |
| `superres/models/{__init__,srcnn,autoencoder_sr,srgan,swinir}.py` | `models/superresolution/` |
| `superres/utils/{__init__,losses,metrics,image_utils,train_utils}.py` | `utils/superresolution/` |
| `superres/train.py` | `train_superresolution.py` (module header only) |
| `dataloader/{__init__,utils,split,superres_dataset,depth_dataset,domain_adapt_dataset}.py` | `dataloader/` |
| `dataloader/README.md` | `dataloader/README.md` |

Two files at the repository root still exist so that documented commands and imports keep
working, and they contain no logic: `train_superresolution.py` and `dataloader/__init__.py`
re-export from here.

**One import stops working**, and it is the only one: `from models.superresolution import
SRCNN` (and the other three nets). `models/` is gone, and the path is now
`symtrs.legacy.superres.models`. Nothing in this repository, in `configs/`, in `tests/` or
in the READMEs ever used it, so the only exposure is a script of the author's kept outside
this repository — written down here so that finding it is a lookup rather than a mystery.
`from dataloader import ...` is the opposite case: it keeps working through the root shim.

`utils/genai/` did **not** move: nothing imports it, and it stays where the author left it
pending a decision about whether it is still wanted. The depth and domain-adaptation
datasets did move, with the rest of `dataloader/`.

`pytorch-CycleGAN-and-pix2pix/` is not part of this tree and never was. It stays at the
repository root, byte-identical to upstream at `2a7afba`, because that byte-identity is the
claim behind every domain-adaptation number.

## Verifying the move

`$BEFORE` below is the commit the move is recorded in, so `HEAD` while the move is still
uncommitted and `HEAD^` once it is:

```bash
BEFORE=HEAD          # or HEAD^ once the move has been committed
```

**1. Byte-identity, from the digests taken before anything moved.** This one works whether
or not the move has been committed, because the "before" side is a file rather than a
commit. The digests are compared as one sorted set, not path by path: every path changed by
definition, and the question is whether any *file* changed.

```bash
grep -v ' train_superresolution.py$' /tmp/symtrs_move/before/sha256.txt \
    | awk '{print $1}' | sort > /tmp/symtrs_move/before_digests.txt
find symtrs/legacy/superres/models symtrs/legacy/superres/utils symtrs/legacy/dataloader \
    -type f \( -name '*.py' -o -name '*.md' \) \
    | xargs shasum -a 256 | awk '{print $1}' | sort > /tmp/symtrs_move/after_digests.txt
diff /tmp/symtrs_move/before_digests.txt /tmp/symtrs_move/after_digests.txt && echo IDENTICAL
```

Expected: 17 lines either side, and `IDENTICAL`.

Comparing per *basename* instead, which is the obvious way to write this, cannot work:
`__init__.py` appears three times (`models/`, `utils/`, `dataloader/`), so a `grep` for it
returns three digests whatever the files contain, and every one of them reads as changed.

**2. The same claim, checked against git.** The 17 files that are not the trainer were
unmodified relative to `HEAD` when they moved — they were tracked and clean — so the blob
each one came from is still in git and can be compared directly:

```bash
git show $BEFORE:models/superresolution/srcnn.py | shasum -a 256   # old
shasum -a 256 symtrs/legacy/superres/models/srcnn.py               # new
```

**3. The rename, with no content change**, shows as `R100` for every file except the
trainer — the one file the move is allowed to touch, and only in its header:

```bash
git diff -M --find-renames=100% --stat $BEFORE
git log --follow --oneline symtrs/legacy/superres/models/srcnn.py
```

The `--follow` line needs the move committed to have anything to follow; until then the
rename is in the working tree and `git status` is the place to look.

**4. The task still produces the number it produced before the move.** A relocation that
changes a metric has changed something it claimed not to:

```bash
diff /tmp/symtrs_move/before/test_metrics.json /tmp/symtrs_move/after/test_metrics.json
```
