"""Co-training: several runs sharing one pass over one loader.

The reason the feature exists is that a 2048px sample costs ~0.2 s of CPU to decode and
~0.15 s of GPU for a depth model on an A6000, so a solo run is data-bound. The tests below
protect the two things that make the sharing safe to report as a number:

  * the guard - a group whose members disagree about the data or the schedule is refused,
    because a member trained on another member's crops would still train and still score;
  * the equivalence - each member gets its own optimiser, checkpoints, CSV rows and test
    pass, `summary.json` says what it shared a pass with, and the loader is read once.

No dataset and no weights: the factory, the model builder and `evaluate` are replaced, as
in `test_trainer_schedule.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import REPO_ROOT, requires_torch

torch = pytest.importorskip("torch", reason="torch is not installed")

from symtrs.config import ConfigError, RunConfig  # noqa: E402
from symtrs.engine import loop, trainer  # noqa: E402


class _TinySeg(torch.nn.Module):
    """The smallest model the segmentation loop accepts: one 1x1 convolution."""

    task = "segmentation"
    key = "tiny"

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 6, 1)

    def preprocess_image(self, image):
        return image

    def forward(self, image):
        return self.conv(image)

    def compute_loss(self, logits, mask):
        loss = torch.nn.functional.cross_entropy(logits, mask, ignore_index=255)
        return loss, {}

    def describe(self) -> str:
        return "tiny"


class _CountingLoader(list):
    """A loader that records how many times it was iterated."""

    def __init__(self, batches):
        super().__init__(batches)
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        return super().__iter__()


def _loader(batches: int, batch: int = 2):
    generator = torch.Generator().manual_seed(0)
    return _CountingLoader([
        {
            "image": torch.rand(batch, 3, 8, 8, generator=generator),
            "mask": torch.randint(0, 6, (batch, 8, 8), generator=generator),
        }
        for _ in range(batches)
    ])


def _cfg(tmp_path: Path, name: str, **train) -> RunConfig:
    cfg = RunConfig()
    cfg.task = "segmentation"
    cfg.model = "tiny"
    cfg.out_dir = str(tmp_path / name)
    cfg.data.root = str(tmp_path)  # never read: the data factory is replaced
    cfg.data.num_workers = 0
    cfg.train.device = "cpu"
    cfg.train.amp = False
    cfg.train.lr = 1e-2
    cfg.train.save_samples_every = 0
    cfg.train.log_every = 0
    cfg.train.warmup_epochs = 0
    cfg.eval.num_visualizations = 0
    for key, value in train.items():
        setattr(cfg.train, key, value)
    return cfg


# ------------------------------------------------------------------- the guard


def test_matching_configs_have_no_conflicts(tmp_path):
    a = _cfg(tmp_path, "a", epochs=2, batch_size=2)
    b = _cfg(tmp_path, "b", epochs=2, batch_size=2)
    assert trainer.group_conflicts([a, b]) == []


def test_an_effective_batch_that_does_not_divide_the_window_is_refused_by_name(tmp_path):
    """3 samples per step cannot be cut out of an 8-sample shared batch in whole steps."""
    conflicts = trainer.group_conflicts([
        _cfg(tmp_path, "a", batch_size=3),
        _cfg(tmp_path, "b", batch_size=8),
    ])
    assert len(conflicts) == 1
    assert conflicts[0].startswith("tiny's effective batch 3 (batch_size x grad_accum) "
                                   "does not divide the group's window 8")


def test_an_effective_batch_that_divides_the_window_shares_it(tmp_path):
    """DeepLabV3 (16 per step) beside SegFormer (32) at 512: the window is 32 and DeepLabV3
    steps twice per shared batch, so both keep the samples-per-step their configs describe."""
    a = _cfg(tmp_path, "a", batch_size=2)
    b = _cfg(tmp_path, "b", batch_size=4, grad_accum=2)
    assert trainer.group_conflicts([a, b]) == []
    assert trainer.shared_window([a, b]) == 8


def test_the_same_effective_batch_may_be_split_differently(tmp_path):
    """`4x2` and `8x1` both average over 8 samples per optimiser step, so they can share one
    window; how those 8 are cut into forward passes is each member's memory choice."""
    a = _cfg(tmp_path, "a", batch_size=4, grad_accum=2)
    b = _cfg(tmp_path, "b", batch_size=8, grad_accum=1)
    assert trainer.group_conflicts([a, b]) == []


def test_members_may_want_different_epochs(tmp_path):
    """A member that finishes early leaves the pass; it does not hold the others back and is
    not retrained to their length."""
    assert trainer.group_conflicts([
        _cfg(tmp_path, "a", epochs=2),
        _cfg(tmp_path, "b", epochs=3),
    ]) == []


def test_a_different_depth_scale_is_refused_even_though_nothing_would_crash(tmp_path):
    """The shared loader builds labels once, for the first member. A member configured for a
    different unit would train on the first member's metres and simply report a worse
    number, so disagreement here has to be an error, not a warning."""
    a = _cfg(tmp_path, "a")
    b = _cfg(tmp_path, "b")
    a.data.depth_scale = 0.01
    b.data.depth_scale = 1.0
    conflicts = trainer.group_conflicts([a, b])
    assert any(c.startswith("data.depth_scale differs") for c in conflicts)


def test_two_members_cannot_share_a_run_dir(tmp_path):
    a = _cfg(tmp_path, "same")
    b = _cfg(tmp_path, "same")
    conflicts = trainer.group_conflicts([a, b])
    assert any("same run dir" in c for c in conflicts)


@pytest.mark.parametrize(
    "path, value",
    [
        ("seed", 7),
        ("data.limit_val_batches", 5),
        ("data.max_samples", 100),
        ("train.patch_size", 128),
    ],
)
def test_every_setting_that_shapes_the_shared_window_is_guarded(tmp_path, path, value):
    """One loader for the group, built from the first member: anything it or the shared
    window depends on has to agree, or a member quietly trains and validates on the first
    member's settings."""
    a = _cfg(tmp_path, "a")
    b = _cfg(tmp_path, "b")
    if "." in path:
        section, field = path.split(".")
        setattr(getattr(b, section), field, value)
    else:
        setattr(b, path, value)
    conflicts = trainer.group_conflicts([a, b])
    assert any(c.startswith(path) for c in conflicts), (path, conflicts)


def test_a_subprocess_task_is_refused_and_says_why(tmp_path):
    a = _cfg(tmp_path, "a")
    b = _cfg(tmp_path, "b")
    a.task = b.task = "shadow_removal"
    conflicts = trainer.group_conflicts([a, b])
    assert any("shadow_removal is driven as a subprocess" in c for c in conflicts)


def test_ddp_is_refused(tmp_path):
    conflicts = trainer.group_conflicts([
        _cfg(tmp_path, "a", gpus=2),
        _cfg(tmp_path, "b", gpus=2),
    ])
    assert any("DDP" in c for c in conflicts)


def test_train_group_raises_the_refusal_with_the_fix(tmp_path):
    with pytest.raises(ConfigError, match="cannot co-train tiny, tiny"):
        trainer.train_group([
            _cfg(tmp_path, "a", batch_size=3),
            _cfg(tmp_path, "b", batch_size=8),
        ])


def test_the_command_is_registered(monkeypatch):
    from symtrs.cli import build_parser
    from symtrs.commands import COMMANDS

    assert COMMANDS["train-group"] == "symtrs.commands.train_group"
    args = build_parser().parse_args([
        "train-group", "configs/a.yaml", "configs/b.yaml",
        "--data-root", "/data", "--out-dir", "runs/g", "--set", "train.epochs=3",
    ])
    assert args.configs == ["configs/a.yaml", "configs/b.yaml"]
    assert args.out_dir == "runs/g" and args.set == ["train.epochs=3"]


# ----------------------------------------------------------------- the behaviour


@pytest.fixture
def scripted(monkeypatch):
    """Replace the data factory, the model builder and `evaluate`.

    `val` is consumed one value per validation pass, in order, shared by both members the
    way a real epoch is; `builds` counts how many models were built.
    """

    class Script:
        val: list = []
        builds = 0
        test_seen: list = []
        loader = None
        # The config each training loader was built from, and every micro-step the members
        # actually took: `(run dir name, images, shared step, close_window)`.
        train_loader_cfgs: list = []
        steps: list = []

    script = Script()

    import symtrs.data.factory as factory
    import symtrs.model.wrapper as wrapper

    script.loader = _loader(2)

    def fake_from_config(cfg, *args, train=True, **kwargs):
        if train:
            script.train_loader_cfgs.append(cfg)
        return script.loader

    # One loader object for every call: the point of the test is that the group iterates
    # it once per epoch instead of building one per member.
    monkeypatch.setattr(factory, "from_config", fake_from_config)

    def fake_build(cfg, **kwargs):
        script.builds += 1
        return _TinySeg()

    monkeypatch.setattr(wrapper, "build_from_config", fake_build)

    original_step = loop.Trainee.step

    def recording_step(self, batch, device, epoch, step, batch_count, close_window=False,
                       log=True):
        script.steps.append((Path(self.cfg.out_dir).name, int(batch["image"].shape[0]),
                             step, close_window, log))
        return original_step(self, batch, device, epoch, step, batch_count,
                             close_window=close_window, log=log)

    monkeypatch.setattr(loop.Trainee, "step", recording_step)

    def fake_evaluate(model, loader, cfg, device, align="", num_visualizations=0,
                      vis_dir=None, vis_prefix="eval"):
        if vis_prefix == "test":
            script.test_seen.append(model.conv.weight.detach().clone())
            return {"miou": 0.321}
        return {"miou": script.val.pop(0)}

    monkeypatch.setattr(loop, "evaluate", fake_evaluate)
    return script


def _train_loader():
    """The loader the group actually iterated (the factory's single shared object)."""
    return None


def test_a_group_trains_both_members_once_over_the_same_batches(tmp_path, scripted):
    scripted.val = [0.9, 0.2, 0.4, 0.8]  # epoch 0: a=0.9 b=0.2; epoch 1: a=0.4 b=0.8
    a = _cfg(tmp_path, "a", epochs=2, batch_size=2)
    b = _cfg(tmp_path, "b", epochs=2, batch_size=2)

    loader = scripted.loader
    summaries = trainer.train_group([a, b])

    assert scripted.builds == 2, "one model per member, and no extra rebuild"
    assert scripted.val == [], "two members x two epochs is four validation passes"
    assert loader.iterations == 2, "one shared pass per epoch, not two"

    # The best epoch is each member's own, not the group's: `a` peaked first.
    for summary, expect_best, expect_epoch in zip(summaries, (0.9, 0.8), (0, 1)):
        assert summary["best"] == pytest.approx(expect_best)
        assert summary["best_epoch"] == expect_epoch
        assert summary["test_weights"] == "best"
        assert summary["test"] == {"miou": 0.321}
        assert summary["cogroup"] == ["tiny", "tiny"]
        assert len(summary["history"]) == 2
        run_dir = Path(summary["run_dir"])
        assert (run_dir / "weights" / "best.pt").is_file()
        assert json.loads((run_dir / "summary.json").read_text())["cogroup"] == ["tiny", "tiny"]
        # Every epoch is a row, including the one that was not scored.
        csv_rows = (run_dir / "metrics.csv").read_text().strip().splitlines()
        assert len(csv_rows) == 3, "header + two epochs"

    # The two members are independent models with independent optimisers.
    weights = [torch.load(Path(s["run_dir"]) / "weights" / "last.pt", map_location="cpu",
                          weights_only=False)["model"]["conv.weight"] for s in summaries]
    assert not torch.equal(weights[0], weights[1])


def test_a_window_is_cut_into_each_members_own_micro_batches(tmp_path, scripted):
    """The loader is built at the shared effective batch and each member cuts it itself.

    `a` is 2x2 and `b` is 4x1, so both want 4 samples per optimiser step but arrive at it
    differently. Each shared batch must give `a` two 2-image forwards (one optimiser step at
    the close of the window) and `b` one 4-image forward - and both must consume the whole
    window, not the first member's slice of it.
    """
    scripted.val = [0.5, 0.5]
    scripted.loader = _loader(2, batch=4)
    a = _cfg(tmp_path, "a", epochs=1, batch_size=2, grad_accum=2)
    b = _cfg(tmp_path, "b", epochs=1, batch_size=4, grad_accum=1)

    trainer.train_group([a, b])

    built = scripted.train_loader_cfgs[0]
    assert (built.train.batch_size, built.train.grad_accum) == (4, 1), (
        "the shared loader must yield one whole optimiser window"
    )

    def taken(name):
        return [item for item in scripted.steps if item[0] == name]

    assert [(size, close, log) for _, size, _, close, log in taken("a")] == [
        (2, False, False), (2, True, True), (2, False, False), (2, True, True)
    ], "one progress line per shared batch, not one per micro-batch"
    assert [(size, close, log) for _, size, _, close, log in taken("b")] == [
        (4, True, True), (4, True, True)
    ]
    assert sum(size for _, size, _, _, _ in taken("a")) == 8, "every sample of the epoch"
    assert sum(size for _, size, _, _, _ in taken("b")) == 8


def test_a_member_steps_its_optimiser_once_per_window_not_once_per_odd_shared_batch(tmp_path):
    """The solo rule `(step + 1) % grad_accum == 0` also fired inside group windows, where
    `step` counts shared batches: a 2x2 member took six optimiser (and schedule) steps over
    four windows. In a group only the driver's explicit close may step."""
    cfg = _cfg(tmp_path, "a", batch_size=2, grad_accum=2)
    model = _TinySeg()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    steps = []
    original = optimizer.step
    optimizer.step = lambda *a, **k: (steps.append(1), original(*a, **k))[1]
    trainee = loop.Trainee(model, optimizer, None, cfg, label="a")
    loop.train_one_epoch_many([trainee], _loader(4, batch=4), torch.device("cpu"),
                              windows=True)
    assert len(steps) == 4, "one optimiser step per 4-sample window"


def test_a_member_with_half_the_window_steps_twice_per_shared_batch(tmp_path, scripted,
                                                                    monkeypatch):
    """`a` is 2x1 beside `b` at 2x2: the shared batch is 4, `a` takes two 2-sample windows
    (two optimiser steps) out of it, and its schedule is sized for twice the shared batches -
    the steps per epoch it would have taken alone."""
    sized = {}
    original = trainer.build_scheduler

    def recording(optimizer, cfg, steps_per_epoch):
        sized[Path(cfg.out_dir).name] = steps_per_epoch
        return original(optimizer, cfg, steps_per_epoch)

    monkeypatch.setattr(trainer, "build_scheduler", recording)
    scripted.val = [0.5, 0.5]
    scripted.loader = _loader(3, batch=4)
    a = _cfg(tmp_path, "a", epochs=1, batch_size=2, grad_accum=1)
    b = _cfg(tmp_path, "b", epochs=1, batch_size=2, grad_accum=2)

    trainer.train_group([a, b])

    built = scripted.train_loader_cfgs[0]
    assert built.train.batch_size == 4, "the loader yields the largest window"
    taken_a = [(size, close) for name, size, _, close, _ in scripted.steps if name == "a"]
    taken_b = [(size, close) for name, size, _, close, _ in scripted.steps if name == "b"]
    assert taken_a == [(2, True), (2, True)] * 3, "two whole windows per shared batch"
    assert taken_b == [(2, False), (2, True)] * 3, "one window of two micro-batches"
    assert sized == {"a": 6, "b": 3}


def test_a_micro_batch_inside_a_window_prints_no_progress_line(tmp_path, capsys):
    """`log=False` must reach the print, not just the signature.

    The driver passed `log=index == last` while `Trainee.step` ignored it, so a 4x2 member
    printed every `epoch N step K` line twice (seen in the first 2048 group's log). The
    driver is exercised directly here so the assertion is on what reaches stdout.
    """
    cfg = _cfg(tmp_path, "a", batch_size=2, grad_accum=2, log_every=1)
    model = _TinySeg()
    trainee = loop.Trainee(model, torch.optim.SGD(model.parameters(), lr=1e-2), None, cfg,
                           label="a")
    loop.train_one_epoch_many([trainee], _loader(3, batch=4), torch.device("cpu"),
                              windows=True)

    lines = [line for line in capsys.readouterr().out.splitlines() if "[a] epoch" in line]
    assert [line.split(" loss=")[0] for line in lines] == [
        "[symtrs] [a] epoch 0 step 1/3", "[symtrs] [a] epoch 0 step 2/3",
        "[symtrs] [a] epoch 0 step 3/3",
    ], "one progress line per shared batch, not one per micro-batch"


def test_the_shortest_member_leaves_but_the_others_keep_training(tmp_path, scripted):
    """Co-training must not stretch one member's schedule to another's: each stops at its
    own `epochs`, keeping the history and best score it earned."""
    scripted.val = [0.9, 0.1, 0.8, 0.7, 0.6]
    a = _cfg(tmp_path, "a", epochs=2, batch_size=2)
    b = _cfg(tmp_path, "b", epochs=3, batch_size=2)

    summaries = trainer.train_group([a, b])

    assert [row["epoch"] for row in summaries[0]["history"]] == [0, 1]
    assert [row["epoch"] for row in summaries[1]["history"]] == [0, 1, 2]
    assert summaries[0]["best"] == pytest.approx(0.9) and summaries[0]["best_epoch"] == 0
    assert summaries[1]["best"] == pytest.approx(0.7) and summaries[1]["best_epoch"] == 1
    assert scripted.loader.iterations == 3, "one shared pass per epoch, not one per member"


def test_a_group_member_scored_nothing_against_a_solo_member_that_also_did_not(tmp_path, scripted):
    """A NaN validation still cannot become "best", in a group as much as alone."""
    scripted.val = [float("nan"), 0.5, float("nan"), 0.5]
    a = _cfg(tmp_path, "a", epochs=2, batch_size=2)
    b = _cfg(tmp_path, "b", epochs=2, batch_size=2)
    summaries = trainer.train_group([a, b])
    assert summaries[0]["best"] is None and summaries[0]["test_weights"] == "last"
    assert summaries[0].get("non_finite_monitor_epochs") == [0, 1]
    assert not (Path(summaries[0]["run_dir"]) / "weights" / "best.pt").exists()
    assert summaries[1]["best"] == pytest.approx(0.5)


def test_a_resumed_member_keeps_its_history_and_starts_where_it_stopped(tmp_path, scripted):
    """A crash between two members' checkpoints leaves them at different epochs; the group
    must train each from its own checkpoint, not restart the one that is behind."""
    scripted.val = [0.5]
    first = _cfg(tmp_path, "a", epochs=1, batch_size=2)
    trainer.train_group([first])  # a one-member group is a plain run

    scripted.val = [0.7, 0.8, 0.9]
    ahead = _cfg(tmp_path, "a", epochs=2, batch_size=2,
                 resume=str(tmp_path / "a" / "weights" / "last.pt"))
    fresh = _cfg(tmp_path, "b", epochs=2, batch_size=2)
    summaries = trainer.train_group([ahead, fresh])

    assert [row["epoch"] for row in summaries[0]["history"]] == [0, 1]
    assert [row["epoch"] for row in summaries[1]["history"]] == [0, 1]
    # `a` trained one more epoch (1); `b` trained both. Three validation passes in total.
    assert scripted.val == []


def test_a_single_config_is_just_a_run(tmp_path, scripted):
    scripted.val = [0.5]
    summary = trainer.train_group([_cfg(tmp_path, "a", epochs=1, batch_size=2)])
    assert summary[0]["test_weights"] == "best"
    assert "cogroup" not in summary[0], "one member shares nothing"


def test_validation_reads_the_spread_subset_and_the_whole_split_by_default(tmp_path, monkeypatch):
    """`train.val_subset` shrinks the passes that pick best.pt (a 2048 pass over 3000 images
    took ~30 min per model); 0 must keep the whole split, and the subset must be the evenly
    spread one, so best.pt is chosen on the same images every epoch."""
    import symtrs.data.factory as factory
    from torch.utils.data import DataLoader, TensorDataset

    full = TensorDataset(torch.arange(50))
    monkeypatch.setattr(factory, "from_config",
                        lambda cfg, *a, **k: DataLoader(full, batch_size=1))

    cfg = _cfg(tmp_path, "a")
    assert len(trainer.validation_loader(cfg).dataset) == 50

    cfg.train.val_subset = 10
    picked = [int(x[0]) for x in trainer.validation_loader(cfg)]
    assert picked == [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]


def test_members_must_agree_on_the_validation_subset(tmp_path):
    a = _cfg(tmp_path, "a")
    b = _cfg(tmp_path, "b")
    b.train.val_subset = 300
    assert any(c.startswith("train.val_subset differs") for c in trainer.group_conflicts([a, b]))


def test_an_unknown_precision_is_refused():
    cfg = _cfg(Path("."), "a")
    cfg.train.amp_dtype = "fp16"
    with pytest.raises(ConfigError, match="train.amp_dtype"):
        cfg.validate()


def test_a_training_step_autocasts_in_the_configured_dtype(tmp_path, monkeypatch):
    """The setting has to reach autocast, not only the config: float16 is the default and
    MiDaS BEiT/Swin-V2 overflowed it."""
    seen = []
    original = loop.autocast_context

    def recording(device, enabled, dtype="float16"):
        seen.append(dtype)
        return original(device, enabled, dtype)

    monkeypatch.setattr(loop, "autocast_context", recording)
    cfg = _cfg(tmp_path, "a")
    cfg.train.amp_dtype = "bfloat16"
    model = _TinySeg()
    trainee = loop.Trainee(model, torch.optim.SGD(model.parameters(), lr=1e-2), None, cfg)
    loop.train_one_epoch_many([trainee], _loader(1), torch.device("cpu"))
    assert seen == ["bfloat16"]
