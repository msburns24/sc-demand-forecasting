import matplotlib as mpl
import matplotlib.pyplot as plt

from catppuccin.extras.matplotlib import CATPPUCCIN_STYLE_DIRECTORY


def configure_matplotlib() -> None:
    styles = mpl.style.core.read_style_directory(CATPPUCCIN_STYLE_DIRECTORY)  # type: ignore[attr-defined]
    mpl.style.core.update_nested_dict(mpl.style.library, styles)  # type: ignore[attr-defined]
    plt.style.use("latte")
