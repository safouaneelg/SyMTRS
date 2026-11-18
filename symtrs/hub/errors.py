"""Errors the hub front-end raises.

They subclass `ValueError` on purpose: `symtrs.cli` already reports `ValueError` as a
one-line user error, so a bad selection or a manifest that disagrees with the package stops
with a message instead of a traceback, without adding another entry to that list.

Every message names the fix (`symtrs get --list`, the install command for an extra), because
these are the errors a first-time user meets: the selection was misspelt, the release moved,
the optional backend is missing, or the disk is too small.
"""

from __future__ import annotations


class HubError(ValueError):
    """A problem with the hub release, the selection or the download environment."""


class ManifestError(HubError):
    """The published manifest is missing, unreadable, or not a schema this package knows."""


class SelectionError(HubError):
    """The requested groups, splits, models or presets do not exist in the manifest."""


class BackendError(HubError):
    """The release could not be reached, or an optional download backend is not installed."""


class DiskSpaceError(HubError):
    """The destination cannot hold the plan."""


__all__ = [
    "BackendError",
    "DiskSpaceError",
    "HubError",
    "ManifestError",
    "SelectionError",
]
