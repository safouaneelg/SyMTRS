"""The release's file-name grammar, in one place.

Every exported file is named

    <modality>_<light>_<resolution>_<key>.<ext>

and the fourteen files of one view share `<key>`, which is what lets a task table pair an
input with its target. `scripts/build_metadata.py` parses names into `views.csv`, and a
release publisher will need the same parse to build task tables; two copies of this grammar
would drift and silently relabel a release, so it lives in the package and is imported.

Nothing here reads an image. A name is a manifest, not a measurement.
"""

from __future__ import annotations

import re
from typing import Optional

#: `<modality>_<light>_<resolution>_loc<NNN>_rot<NNN>deg_h<NN>_x<sign><cm>_y<...>_z<...>`.
#: The engine position is stored in centimetres; `parse_view_name` reports metres.
VIEW_NAME = re.compile(
    r"^(?P<modality>rgb|depth|seg|semantic)_(?P<light>day|night|noshadow)_(?P<resolution>\d+)_"
    r"(?P<key>loc(?P<location>\d+)_rot(?P<rotation>\d+)deg_h(?P<altitude>\d+)_"
    r"x(?P<x>[+-]\d+)_y(?P<y>[+-]\d+)_z(?P<z>[+-]\d+))$"
)

#: The fourteen files a complete view carries: one native render per lighting state, three
#: low-resolution day renders, four depth maps, four masks (Sect. 3 of the manuscript).
EXPECTED_VIEW_FILES = frozenset(
    {
        ("rgb", "day", "2048"),
        ("rgb", "day", "1024"),
        ("rgb", "day", "512"),
        ("rgb", "day", "256"),
        ("rgb", "night", "2048"),
        ("rgb", "noshadow", "2048"),
        ("depth", "day", "2048"),
        ("depth", "day", "1024"),
        ("depth", "day", "512"),
        ("depth", "day", "256"),
        ("seg", "day", "2048"),
        ("seg", "day", "1024"),
        ("seg", "day", "512"),
        ("seg", "day", "256"),
    }
)

#: `seg` is also exported as `semantic` in some trees; either name fills the mask slots.
MASK_ALIASES = frozenset({"seg", "semantic"})


def parse_view_name(stem: str) -> Optional[dict]:
    """The fields of one export name, or None when the name is not a SyMTRS export.

    `x`, `y` and `z` are metres: the release stores centimetres, as the engine does.
    """
    match = VIEW_NAME.match(stem)
    if match is None:
        return None
    fields = match.groupdict()
    for axis in ("x", "y", "z"):
        fields[axis] = int(fields[axis]) / 100.0
    return fields


#: Kept as the name `build_metadata.py` used before the grammar moved here.
NAME = VIEW_NAME


__all__ = [
    "EXPECTED_VIEW_FILES",
    "MASK_ALIASES",
    "NAME",
    "VIEW_NAME",
    "parse_view_name",
]
