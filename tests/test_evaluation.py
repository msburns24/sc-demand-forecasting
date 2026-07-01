import math

import pandas as pd
import pytest

from src.evaluation import segment_metrics


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
