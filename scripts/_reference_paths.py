"""Where the MMSA reference checkout and its shim live on this machine.

Both were hard-coded to ``/workspace``, a path that existed on the first machine
this project ran on and on neither of the two since. A wrong path here does not
fail loudly: ``check_almt_equivalence.py`` prints SKIP and exits 0, so
convention 5's gate does not turn red when it disappears -- it just stops
existing. That has now happened twice, for two different reasons (the checkout
gone after a machine move, and before that a shim path pointing into a scratch
directory that did not outlive the session that made it). So the path stops
being a constant.

Resolution order, first hit wins:

1. ``$MMSA_DIR`` / ``$MMSA_SHIM`` -- an explicit override, unchanged behaviour
   for anything that already sets them.
2. ``<repo>/.mmsa-reference/`` -- the default. Gitignored, machine-local, and
   found with nothing exported, which is the property the old default lacked.
3. ``/workspace/MMSA`` and ``/workspace/mmsa_env/shim`` -- where the first two
   machines kept it. Only used if actually present.

Nothing here creates directories; ``scripts/setup_mmsa_reference.sh`` does that,
and resolves the same three candidates in the same order.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
#: Default root for both halves of the reference. Gitignored.
LOCAL_ROOT = REPO / ".mmsa-reference"
#: Where the RTX 5070 Ti and RTX 5080 machines put it.
LEGACY_MMSA = Path("/workspace/MMSA")
LEGACY_SHIM = Path("/workspace/mmsa_env/shim")


def mmsa_checkout() -> Path:
    """The MMSA git checkout. May not exist -- callers report SKIP if so."""
    override = os.environ.get("MMSA_DIR")
    if override:
        return Path(override)
    local = LOCAL_ROOT / "MMSA"
    if local.exists():
        return local
    if LEGACY_MMSA.exists():
        return LEGACY_MMSA
    return local


def mmsa_src() -> Path:
    """MMSA's ``src`` directory, which is what goes on ``sys.path``."""
    return mmsa_checkout() / "src"


def mmsa_shim() -> Path:
    """The directory of symlinks holding transformers 4.x, einops and easydict."""
    override = os.environ.get("MMSA_SHIM")
    if override:
        return Path(override)
    local = LOCAL_ROOT / "env" / "shim"
    if local.exists():
        return local
    if LEGACY_SHIM.exists():
        return LEGACY_SHIM
    return local
