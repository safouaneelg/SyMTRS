"""Every `symtrs <command>` written into a message must name a real command.

Error text is the one part of a CLI read by someone already stuck, and a hint that does not
run is worse than no hint: it costs a round trip and suggests the tool is broken. Two of
these were live. `MissingDirectoryError` sent the reader to `symtrs inspect --root`, but
`inspect`'s only argument is a positional checkpoint, so the flag was rejected - and that is
the message a wrong `--data-root` produces, i.e. the first thing a new user sees. The
no-evaluation-loop error named `symtrs metrics domain`, a subcommand that never existed.
Both were found by reading them, not by running them, which is why this is a static check:
these messages sit on paths that need a dataset, a GPU and a download to reach.

A mention counts in one of two shapes. Inside backticks it is certainly an invocation, and
what follows is free to be an argument or an English word - the original bug read
``run `symtrs metrics domain` for ...``, so requiring an argument there would miss exactly
the case this file exists for. Otherwise the mention must be followed by an argument: a
flag, a placeholder, a config file, or the end of the line. That second, narrower shape is
what keeps prose such as `a readable symtrs checkpoint` and the import form
`from symtrs import x` from being read as command names.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parent.parent / "symtrs"

#: A command named inside a backtick span - always an invocation, whatever follows.
_BACKTICKED = re.compile(r"`symtrs ([a-z][a-z-]*)")

#: A bare command followed by an argument shape: a flag, a placeholder, a number, a quoted
#: value, a file, or the end of the line.
_BARE = re.compile(
    r"symtrs ([a-z][a-z-]*)"
    r"(?=\s+(?:--?[a-z]|<|\d|['\"`]|\S+\.(?:ya?ml|pt|pth|png|json|txt))|\s*$)"
)

#: `symtrs train_sr_x2.yaml` is the bare-YAML shorthand that the entry point rewrites to
#: `train`, so the name after it is a file, not a command.
_NOT_COMMANDS = {"train_sr_x"}


def _real_commands() -> set:
    from symtrs.cli import build_parser

    parser = build_parser()
    for action in parser._subparsers._group_actions:
        return set(action.choices)
    raise AssertionError("no subparsers registered; the CLI would accept nothing")


def _names_in(line: str) -> list:
    """The command names a single source line mentions, in order and without repeats.

    Deduplicated because the two patterns overlap by design: `` `symtrs inspect --root x` ``
    is both a backticked mention and a bare one, and reporting the same broken command twice
    would read as two findings.
    """
    names = _BACKTICKED.findall(line) + _BARE.findall(line)
    return [name for name in dict.fromkeys(names) if name not in _NOT_COMMANDS]


def _mentions():
    """Every command invocation mentioned in the package, with where it came from."""
    found = []
    for path in sorted(SOURCE.rglob("*.py")):
        relative = path.relative_to(SOURCE.parent)
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            found.extend((name, f"{relative}:{number}") for name in _names_in(line))
    return found


def test_the_command_list_is_not_empty():
    """Guards the guard: with no commands registered, every mention below would fail for
    the wrong reason and the reported message would be nonsense."""
    assert len(_real_commands()) >= 8


def test_the_scan_finds_the_mentions_it_is_supposed_to():
    """Guards the other end. The patterns above are narrow on purpose, and one that quietly
    stopped matching would make the test below pass by finding nothing at all."""
    found = _mentions()
    assert len(found) >= 40, f"only {len(found)} mentions found; has the pattern drifted?"


@pytest.mark.parametrize(
    "line,expected",
    [
        # The two shapes that were live bugs, written as they appeared.
        ('f"reported through `symtrs metrics domain`"', ["metrics"]),
        ("f'run `symtrs inspect --root <dataset root>` to see'", ["inspect"]),
        # A bare invocation with an argument, and one that ends the line.
        ("print('try: symtrs train --task depth')", ["train"]),
        ("    symtrs doctor", ["doctor"]),
        # Not invocations: the import form, and prose.
        ("from symtrs import CONFIG_FORMAT_VERSION", []),
        ('f"{source!r} is not a readable symtrs checkpoint"', []),
        # A file, not a command.
        ("  symtrs train_sr_x2.yaml", []),
    ],
)
def test_the_scan_recognises_invocations_and_ignores_prose(line, expected):
    """Pinned on the exact lines, because the whole value of this file is that the pattern
    is neither too loose (prose becomes a phantom command) nor too tight (the real bug
    slips through). `symtrs metrics domain` in particular is followed by an English word
    and is only recognisable by its backticks."""
    names = _names_in(line)
    assert names == expected


def test_every_command_named_in_a_message_exists():
    commands = _real_commands()
    offenders = [
        f"{where} mentions `symtrs {name}`" for name, where in _mentions() if name not in commands
    ]

    assert not offenders, (
        "these messages point at commands that do not exist:\n  "
        + "\n  ".join(sorted(set(offenders)))
        + f"\nreal commands: {', '.join(sorted(commands))}"
    )
