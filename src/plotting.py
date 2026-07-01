import importlib.util
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

# Catppuccin flavour used across the project's charts.
_FLAVOR = "latte"
_FALLBACK_STYLE = "seaborn-v0_8-whitegrid"
_FALLBACK_CMAP = "viridis"


def _catppuccin_style_path(flavor: str = _FLAVOR) -> Path | None:
    """
    Locate a Catppuccin `.mplstyle` file without importing `catppuccin`.

    The `catppuccin` package (2.5.x) registers its styles at import time via
    `matplotlib.style.core`, which matplotlib >= 3.11 removed — so `import
    catppuccin` raises `AttributeError`. `find_spec` locates the installed
    package directory *without* executing its `__init__`, so we can load the
    shipped style file directly. Returns `None` if it isn't installed.
    """
    spec = importlib.util.find_spec("catppuccin")
    if spec is None or not spec.submodule_search_locations:
        return None
    path = (
        Path(spec.submodule_search_locations[0])
        / "extras"
        / "matplotlib_styles"
        / f"{flavor}.mplstyle"
    )
    return path if path.exists() else None


def configure_matplotlib(flavor: str = _FLAVOR) -> None:
    """
    Apply the project's Catppuccin plot style.

    Loads the shipped `<flavor>.mplstyle` directly (see
    `_catppuccin_style_path` for why we avoid `import catppuccin`), falling
    back to a clean built-in style if it can't be found — so this never raises.
    """
    style_path = _catppuccin_style_path(flavor)
    plt.style.use(str(style_path) if style_path is not None else _FALLBACK_STYLE)
    # The Catppuccin styles set `image.cmap` to a custom colormap that only the
    # (bypassed) `import catppuccin` registers, which would make imshow/colorbar
    # raise. Reset it to a valid default when that colormap isn't registered.
    if plt.rcParams["image.cmap"] not in matplotlib.colormaps:
        plt.rcParams["image.cmap"] = _FALLBACK_CMAP
