import math

import numpy as np
import pandas as pd
import pytest

from src.evaluation import compute_error_stats, segment_metrics


def make_preds():
    """Two SKUs, two val days each. I1 -> AX (small error), I2 -> CZ (has a zero)."""
    rows = [
        ("S1", "I1", "2020-01-01", 10, 12),  # err 2
        ("S1", "I1", "2020-01-02", 20, 18),  # err 2
        ("S1", "I2", "2020-01-01", 0, 1),    # zero actual -> excluded from MAPE
        ("S1", "I2", "2020-01-02", 4, 2),    # err 2
    ]
    return pd.DataFrame(rows, columns=["store_id", "item_id", "date", "sales", "prediction"])


def make_segments():
    return pd.DataFrame(
        {"store_id": ["S1", "S1"], "item_id": ["I1", "I2"], "abc_xyz": ["AX", "CZ"]}
    )


class TestSegmentMetrics:
    def test_per_cell_rmse_mape(self):
        m = segment_metrics(make_preds(), make_segments()).set_index("abc_xyz")
        # AX: errors [2,2] -> rmse 2; mape mean(|2/10|,|2/20|)=15%
        assert m.loc["AX", "rmse"] == pytest.approx(2.0)
        assert m.loc["AX", "mape"] == pytest.approx(15.0)
        # CZ: errors [1,2] -> rmse sqrt(2.5); mape only the non-zero row (|2/4|=50%)
        assert m.loc["CZ", "rmse"] == pytest.approx(math.sqrt(2.5), abs=1e-3)
        assert m.loc["CZ", "mape"] == pytest.approx(50.0)

    def test_counts(self):
        m = segment_metrics(make_preds(), make_segments()).set_index("abc_xyz")
        assert m.loc["AX", "n_skus"] == 1
        assert m.loc["AX", "n_obs"] == 2

    def test_one_row_per_cell(self):
        m = segment_metrics(make_preds(), make_segments())
        assert len(m) == m["abc_xyz"].nunique()

    def test_multi_sku_cell_counts(self):
        preds = pd.concat(
            [
                make_preds(),
                pd.DataFrame(
                    [("S1", "I3", "2020-01-01", 5, 5), ("S1", "I3", "2020-01-02", 5, 5)],
                    columns=["store_id", "item_id", "date", "sales", "prediction"],
                ),
            ],
            ignore_index=True,
        )
        segs = pd.concat(
            [make_segments(), pd.DataFrame({"store_id": ["S1"], "item_id": ["I3"], "abc_xyz": ["AX"]})],
            ignore_index=True,
        )
        m = segment_metrics(preds, segs).set_index("abc_xyz")
        assert m.loc["AX", "n_skus"] == 2
        assert m.loc["AX", "n_obs"] == 4


class TestComputeErrorStats:
    def test_bias_sign_convention(self):
        # under-forecast: y_true > y_pred -> positive bias
        under = compute_error_stats([10, 10, 10], [5, 5, 5], ["AX"] * 3)
        assert under.loc[0, "bias"] > 0
        # over-forecast: y_true < y_pred -> negative bias
        over = compute_error_stats([5, 5, 5], [10, 10, 10], ["AX"] * 3)
        assert over.loc[0, "bias"] < 0

    def test_bias_and_sigma_values(self):
        # residuals [-2, 2] for AX, [-1, 2] for CZ (mirrors make_preds() errors)
        y_true, y_pred = [10, 20, 0, 4], [12, 18, 1, 2]
        segs = ["AX", "AX", "CZ", "CZ"]
        out = compute_error_stats(y_true, y_pred, segs).set_index("abc_xyz")
        assert out.loc["AX", "bias"] == pytest.approx(0.0)
        assert out.loc["AX", "sigma"] == pytest.approx(np.std([-2, 2], ddof=1), abs=1e-4)
        assert out.loc["CZ", "bias"] == pytest.approx(0.5)

    def test_one_row_per_cell(self):
        out = compute_error_stats([1, 2, 3], [1, 1, 1], ["AX", "AX", "CZ"])
        assert len(out) == 2

    def test_n_obs_counts(self):
        out = compute_error_stats(
            [1, 2, 3, 4], [1, 1, 1, 1], ["AX", "AX", "AX", "CZ"]
        ).set_index("abc_xyz")
        assert out.loc["AX", "n_obs"] == 3
        assert out.loc["CZ", "n_obs"] == 1

    def test_single_obs_cell_sigma_skew_shapiro_are_nan(self):
        out = compute_error_stats(
            [1, 5, 5, 5], [0, 2, 3, 4], ["CZ", "AX", "AX", "AX"]
        ).set_index("abc_xyz")
        assert pd.isna(out.loc["CZ", "sigma"])
        assert pd.isna(out.loc["CZ", "skew"])
        assert pd.isna(out.loc["CZ", "shapiro_p"])
        assert pd.isna(out.loc["CZ", "is_normal"])

    def test_two_obs_cell_has_sigma_but_not_skew_or_shapiro(self):
        out = compute_error_stats([1, 2], [0, 1], ["AX", "AX"]).set_index("abc_xyz")
        assert not pd.isna(out.loc["AX", "sigma"])
        assert pd.isna(out.loc["AX", "skew"])
        assert pd.isna(out.loc["AX", "shapiro_p"])

    def test_constant_residuals_zero_variance_is_safe(self):
        # 5 identical residuals -> sigma 0, shapiro undefined, no crash/NaN-warning leak
        out = compute_error_stats([5] * 5, [3] * 5, ["AX"] * 5).set_index("abc_xyz")
        assert out.loc["AX", "sigma"] == pytest.approx(0.0)
        assert pd.isna(out.loc["AX", "shapiro_p"])
        assert pd.isna(out.loc["AX", "is_normal"])

    def test_ax_tighter_than_cz(self):
        # AX residuals clustered near 0; CZ residuals spread wide -- the AC's hypothesis
        ax_res = [-1, 0, 1, 0, -1, 1, 0]
        cz_res = [-20, 15, -30, 25, 10, -15, 5]
        y_true = ax_res + cz_res
        y_pred = [0] * len(ax_res) + [0] * len(cz_res)
        segs = ["AX"] * len(ax_res) + ["CZ"] * len(cz_res)
        out = compute_error_stats(y_true, y_pred, segs).set_index("abc_xyz")
        assert out.loc["AX", "sigma"] < out.loc["CZ", "sigma"]

    def test_shapiro_sampling_is_deterministic_and_uses_full_data_for_bias_sigma(self):
        rng = np.random.default_rng(1)
        residuals = rng.normal(0, 1, 200).tolist()
        y_true, y_pred = residuals, [0] * 200
        segs = ["AX"] * 200
        out1 = compute_error_stats(
            y_true, y_pred, segs, shapiro_sample_size=20, random_state=7
        )
        out2 = compute_error_stats(
            y_true, y_pred, segs, shapiro_sample_size=20, random_state=7
        )
        assert out1.loc[0, "shapiro_p"] == out2.loc[0, "shapiro_p"]
        assert out1.loc[0, "bias"] == pytest.approx(np.mean(residuals), abs=1e-4)
        assert out1.loc[0, "sigma"] == pytest.approx(np.std(residuals, ddof=1), abs=1e-4)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            compute_error_stats([1, 2, 3], [1, 2], ["AX", "AX", "AX"])
