"""The repository map the release is planned under.

One HF project per vertical, plus one for the image store and one for the metadata that
carries the catalog. The launcher reads the real URLs from the manifest; these are the
placeholders the publisher, the docs and the tests use until the projects exist, and they
are what `symtrs get --list` shows for a release that publishes its repo map.

Splitting the projects this way is what makes "pull only what you need" true at the
repository level too: choosing depth fetches the depth tables, the shared views those tables
point at, and metadata - not the super-resolution or translation projects.
"""

from __future__ import annotations

from typing import Dict

#: The key whose repo carries `symtrs-hub.json`. It is also the group that is always pulled
#: with a vertical (see `default_groups` in the manifest), so the catalog is never a
#: separate download from the data documentation.
CATALOG_REPO_KEY = "metadata"

#: Placeholders. Change these in the manifest, not in code, once the projects are pushed.
PLACEHOLDER_REPOS: Dict[str, Dict[str, str]] = {
    "metadata": {"repo": "safouaneelg/SyMTRS-metadata", "type": "dataset"},
    "views": {"repo": "safouaneelg/SyMTRS-views", "type": "dataset"},
    "depth": {"repo": "safouaneelg/SyMTRS-depth", "type": "dataset"},
    "sr": {"repo": "safouaneelg/SyMTRS-sr", "type": "dataset"},
    "segmentation": {"repo": "safouaneelg/SyMTRS-segmentation", "type": "dataset"},
    "night_translation": {"repo": "safouaneelg/SyMTRS-night_translation", "type": "dataset"},
    "shadow_removal": {"repo": "safouaneelg/SyMTRS-shadow_removal", "type": "dataset"},
    "models-depth": {"repo": "safouaneelg/SyMTRS-models-depth", "type": "model"},
    "models-sr": {"repo": "safouaneelg/SyMTRS-models-sr", "type": "model"},
    "models-segmentation": {"repo": "safouaneelg/SyMTRS-models-segmentation", "type": "model"},
    "models-night_translation": {
        "repo": "safouaneelg/SyMTRS-models-night_translation",
        "type": "model",
    },
    "models-shadow_removal": {
        "repo": "safouaneelg/SyMTRS-models-shadow_removal",
        "type": "model",
    },
}

#: The default catalog repo (`symtrs get` with no --mirror/--repo).
DEFAULT_CATALOG_REPO = PLACEHOLDER_REPOS[CATALOG_REPO_KEY]["repo"]

#: The repos a full install of one vertical touches, for docs and for `--list` hints.
VERTICAL_REPOS = {
    "depth": ("depth", "models-depth"),
    "segmentation": ("segmentation", "models-segmentation"),
    "domain_adapt": ("night_translation", "models-night_translation"),
    "shadow_removal": ("shadow_removal", "models-shadow_removal"),
    "superresolution": ("sr", "models-sr"),
}


def repo_url(repo: str, repo_type: str = "dataset") -> str:
    """The browser URL for one repository, for the catalogue and the TUI footer."""
    prefix = "datasets/" if repo_type == "dataset" else ""
    return f"https://huggingface.co/{prefix}{repo}"


__all__ = [
    "CATALOG_REPO_KEY",
    "DEFAULT_CATALOG_REPO",
    "PLACEHOLDER_REPOS",
    "VERTICAL_REPOS",
    "repo_url",
]
