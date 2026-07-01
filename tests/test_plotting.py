import matplotlib

matplotlib.use("Agg")  # headless backend for tests
import matplotlib.pyplot as plt  # noqa: E402

from src.plotting import _catppuccin_style_path, configure_matplotlib  # noqa: E402


class TestConfigureMatplotlib:
    def test_runs_without_error(self):
        # Regression: catppuccin 2.5 + matplotlib >= 3.11 used to crash on import.
        configure_matplotlib()

    def test_image_cmap_is_valid(self):
        # The Catppuccin style's `image.cmap: latte` isn't registered here; it
        # must be sanitised so imshow/colorbar don't fail.
        configure_matplotlib()
        assert plt.rcParams["image.cmap"] in matplotlib.colormaps

    def test_applies_catppuccin_latte_when_available(self):
        if _catppuccin_style_path() is not None:
            configure_matplotlib("latte")
            assert plt.rcParams["axes.facecolor"].lower() == "#eff1f5"

    def test_unknown_flavour_falls_back_without_error(self):
        # Missing flavour -> no style file -> built-in fallback, never raises.
        assert _catppuccin_style_path("not_a_flavour") is None
        configure_matplotlib("not_a_flavour")
