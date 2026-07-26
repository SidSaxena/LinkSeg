"""Let dgl import when its graphbolt build does not match the installed torch.

``dgl.graphbolt`` loads ``libgraphbolt_pytorch_<torch version>`` at import time
and raises ``FileNotFoundError`` when that exact file is missing. The wheels
carry builds for a handful of torch versions -- on macOS, 2.1.0 through 2.3.0 --
so any newer torch fails the import before dgl itself is usable.

LinkSeg uses ``dgl.nn``'s EdgeGATConv and never touches graphbolt, so putting an
empty module in its place lets the import proceed. Verified on macOS with dgl
2.2.0 against torch 2.10: EdgeGATConv builds and runs.

Import this before dgl.
"""

import importlib.util
import os
import sys
import types


def _graphbolt_library_missing():
    """True when dgl would raise for want of a matching graphbolt build.

    dgl is located rather than imported: importing it is what triggers the
    failure this module exists to avoid.
    """
    try:
        import torch
    except Exception:
        return False

    version = torch.__version__.split("+", maxsplit=1)[0]
    if sys.platform.startswith("linux"):
        basename = f"libgraphbolt_pytorch_{version}.so"
    elif sys.platform.startswith("darwin"):
        basename = f"libgraphbolt_pytorch_{version}.dylib"
    elif sys.platform.startswith("win"):
        basename = f"graphbolt_pytorch_{version}.dll"
    else:
        return False

    try:
        spec = importlib.util.find_spec("dgl")
    except Exception:
        return False
    if spec is None or not spec.submodule_search_locations:
        return False

    return not any(
        os.path.exists(os.path.join(directory, "graphbolt", basename))
        for directory in spec.submodule_search_locations
    )


def install():
    """Stub dgl.graphbolt when its library is absent. No effect otherwise."""
    if "dgl" in sys.modules or "dgl.graphbolt" in sys.modules:
        return False
    if not _graphbolt_library_missing():
        return False
    stub = types.ModuleType("dgl.graphbolt")
    stub.__doc__ = "Stubbed by dgl_compat: no graphbolt build matches this torch."
    sys.modules["dgl.graphbolt"] = stub
    return True


install()
