import pandas as pd
import pytest

from src.segmentation import classify_abc, classify_mts_mto, classify_xyz


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def abc_df():
    """Two stores, four items each. Per-store revenues: 700, 200, 70, 30 (total 1000).

    With default thresholds (0.70, 0.90):
      cumulative after sort: 0.70, 0.90, 0.97, 1.00
      → I1=A, I2=B, I3=C, I4=C
    """
    rows = []
    for store in ["CA_1", "TX_1"]:
        for date in ["2023-01-01", "2023-01-02"]:
            rows += [
                {"store_id": store, "item_id": "I1", "date": date, "revenue": 350},
                {"store_id": store, "item_id": "I2", "date": date, "revenue": 100},
                {"store_id": store, "item_id": "I3", "date": date, "revenue": 35},
                {"store_id": store, "item_id": "I4", "date": date, "revenue": 15},
            ]
    return pd.DataFrame(rows)


# Weekly dates (Mondays) used across XYZ fixtures.
_WEEKS = ["2023-01-02", "2023-01-09", "2023-01-16", "2023-01-23"]


def _xyz_rows(store, item, demands):
    return [
        {"store_id": store, "item_id": item, "date": date, "demand": d}
        for date, d in zip(_WEEKS, demands)
    ]


@pytest.fixture
def xyz_df():
    """Single store, four items with distinct CV profiles:

    X_item:  [10, 10, 10, 10] → CV = 0         → X
    Y_item:  [2, 10, 18, 10]  → CV ≈ 0.65      → Y
    Z_item:  [0,  0,  0, 20]  → CV = 2.0       → Z
    zero:    [0,  0,  0,  0]  → CV = NaN → inf → Z
    """
    rows = (
        _xyz_rows("S1", "X_item", [10, 10, 10, 10])
        + _xyz_rows("S1", "Y_item", [2, 10, 18, 10])
        + _xyz_rows("S1", "Z_item", [0, 0, 0, 20])
        + _xyz_rows("S1", "zero", [0, 0, 0, 0])
    )
    return pd.DataFrame(rows)


# ── classify_abc tests ────────────────────────────────────────────────────────


class TestClassifyAbc:
    def test_all_items_classified(self, abc_df):
        result = classify_abc(abc_df, "revenue")
        assert result["abc_class"].notna().all()
        assert set(result["abc_class"].unique()).issubset({"A", "B", "C"})

    def test_classification_labels(self, abc_df):
        result = classify_abc(abc_df, "revenue")
        ca1 = result[result["store_id"] == "CA_1"].drop_duplicates("item_id")
        classes = ca1.set_index("item_id")["abc_class"].to_dict()
        assert classes["I1"] == "A"
        assert classes["I2"] == "B"
        assert classes["I3"] == "C"
        assert classes["I4"] == "C"

    def test_a_revenue_within_threshold(self, abc_df):
        """Sum of A-item revenues must not exceed a_threshold of store total."""
        result = classify_abc(abc_df, "revenue")
        for store in result["store_id"].unique():
            store_rows = result[result["store_id"] == store]
            total = store_rows["revenue"].sum()
            a_total = store_rows[store_rows["abc_class"] == "A"]["revenue"].sum()
            assert a_total / total <= 0.70 + 1e-9

    def test_b_revenue_within_threshold(self, abc_df):
        """Cumulative A+B revenue must not exceed b_threshold of store total."""
        result = classify_abc(abc_df, "revenue")
        for store in result["store_id"].unique():
            store_rows = result[result["store_id"] == store]
            total = store_rows["revenue"].sum()
            ab_total = store_rows[store_rows["abc_class"].isin(["A", "B"])][
                "revenue"
            ].sum()
            assert ab_total / total <= 0.90 + 1e-9

    def test_each_store_classified_independently(self, abc_df):
        """Both stores should have the same A/B/C mapping since revenues are identical."""
        result = classify_abc(abc_df, "revenue")
        for item in ["I1", "I2", "I3", "I4"]:
            ca1_class = result[
                (result["store_id"] == "CA_1") & (result["item_id"] == item)
            ]["abc_class"].iloc[0]
            tx1_class = result[
                (result["store_id"] == "TX_1") & (result["item_id"] == item)
            ]["abc_class"].iloc[0]
            assert ca1_class == tx1_class

    def test_preserves_row_count(self, abc_df):
        result = classify_abc(abc_df, "revenue")
        assert len(result) == len(abc_df)

    def test_deterministic_output(self, abc_df):
        result1 = classify_abc(abc_df, "revenue")
        result2 = classify_abc(abc_df, "revenue")
        pd.testing.assert_frame_equal(
            result1.reset_index(drop=True), result2.reset_index(drop=True)
        )

    def test_custom_thresholds(self, abc_df):
        """With a=0.50 threshold I1 (70% of revenue) should become B, not A."""
        result = classify_abc(abc_df, "revenue", a_threshold=0.50, b_threshold=0.85)
        ca1 = result[result["store_id"] == "CA_1"].drop_duplicates("item_id")
        classes = ca1.set_index("item_id")["abc_class"].to_dict()
        assert classes["I1"] == "B"


# ── classify_xyz tests ────────────────────────────────────────────────────────


class TestClassifyXyz:
    def test_all_items_classified(self, xyz_df):
        result = classify_xyz(xyz_df, "demand", "date")
        assert result["xyz_class"].notna().all()
        assert set(result["xyz_class"].unique()).issubset({"X", "Y", "Z"})

    def test_x_item_constant_demand(self, xyz_df):
        result = classify_xyz(xyz_df, "demand", "date")
        x_class = result[result["item_id"] == "X_item"]["xyz_class"].iloc[0]
        assert x_class == "X"

    def test_y_item_moderate_cv(self, xyz_df):
        # Y_item: [2, 10, 18, 10] → mean=10, std≈6.53, CV≈0.65
        result = classify_xyz(xyz_df, "demand", "date")
        y_class = result[result["item_id"] == "Y_item"]["xyz_class"].iloc[0]
        assert y_class == "Y"

    def test_z_item_erratic_demand(self, xyz_df):
        # Z_item: [0, 0, 0, 20] → mean=5, std=10, CV=2.0
        result = classify_xyz(xyz_df, "demand", "date")
        z_class = result[result["item_id"] == "Z_item"]["xyz_class"].iloc[0]
        assert z_class == "Z"

    def test_zero_demand_series_is_z(self, xyz_df):
        """All-zero demand has undefined CV and must fall back to Z."""
        result = classify_xyz(xyz_df, "demand", "date")
        zero_class = result[result["item_id"] == "zero"]["xyz_class"].iloc[0]
        assert zero_class == "Z"

    def test_cv_exactly_at_x_threshold_is_y(self):
        """CV = 0.5 (equal to x_threshold) must classify as Y, not X."""
        # [5, 10, 15] → mean=10, std=5, CV=0.5
        rows = _xyz_rows("S1", "boundary", [5, 10, 15])
        df = pd.DataFrame(rows[: len(rows) - 1])  # use only 3 weeks
        df = pd.DataFrame(_xyz_rows("S1", "boundary", [5, 10, 15])[:3])
        result = classify_xyz(df, "demand", "date")
        assert result["xyz_class"].iloc[0] == "Y"

    def test_cv_exactly_at_y_threshold_is_z(self):
        """CV = 1.0 (equal to y_threshold) must classify as Z, not Y."""
        # [0, 2, 4] → mean=2, std=2, CV=1.0
        df = pd.DataFrame(_xyz_rows("S1", "boundary", [0, 2, 4])[:3])
        result = classify_xyz(df, "demand", "date")
        assert result["xyz_class"].iloc[0] == "Z"

    def test_cv_calculation_explicit(self):
        """Verify CV is computed on weekly-aggregated demand (not raw rows)."""
        # Two rows per week that sum to 10 — should behave identically to one row of 10.
        rows = []
        for date in _WEEKS:
            rows.append({"store_id": "S1", "item_id": "I1", "date": date, "demand": 6})
            rows.append({"store_id": "S1", "item_id": "I1", "date": date, "demand": 4})
        df = pd.DataFrame(rows)
        result = classify_xyz(df, "demand", "date")
        # Weekly totals are all 10 → CV = 0 → X
        assert result["xyz_class"].iloc[0] == "X"

    def test_preserves_row_count(self, xyz_df):
        result = classify_xyz(xyz_df, "demand", "date")
        assert len(result) == len(xyz_df)

    def test_deterministic_output(self, xyz_df):
        result1 = classify_xyz(xyz_df, "demand", "date")
        result2 = classify_xyz(xyz_df, "demand", "date")
        pd.testing.assert_frame_equal(
            result1.reset_index(drop=True), result2.reset_index(drop=True)
        )

    def test_custom_thresholds(self):
        """With x_threshold=0.8, a CV=0.65 item should become X."""
        df = pd.DataFrame(_xyz_rows("S1", "I1", [2, 10, 18, 10]))
        result = classify_xyz(df, "demand", "date", x_threshold=0.8)
        assert result["xyz_class"].iloc[0] == "X"


# ── classify_mts_mto tests ────────────────────────────────────────────────────


class TestClassifyMtsMto:
    _POLICY_MAP = {
        "AX": "MTS",
        "AY": "MTS",
        "AZ": "MTS",
        "BX": "MTS",
        "BY": "MTS",
        "BZ": "MTS/review",
        "CX": "MTS",
        "CY": "MTS/review",
        "CZ": "MTO",
    }

    def test_all_nine_cells(self):
        abc = pd.Series([cell[0] for cell in self._POLICY_MAP])
        xyz = pd.Series([cell[1] for cell in self._POLICY_MAP])
        result = classify_mts_mto(abc, xyz)
        expected = pd.Series(list(self._POLICY_MAP.values()))
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_mts_cells(self):
        for cell in ["AX", "AY", "AZ", "BX", "BY", "CX"]:
            result = classify_mts_mto(pd.Series([cell[0]]), pd.Series([cell[1]]))
            assert result.iloc[0] == "MTS", f"{cell} should be MTS"

    def test_mts_review_cells(self):
        for cell in ["BZ", "CY"]:
            result = classify_mts_mto(pd.Series([cell[0]]), pd.Series([cell[1]]))
            assert result.iloc[0] == "MTS/review", f"{cell} should be MTS/review"

    def test_cz_is_mto(self):
        result = classify_mts_mto(pd.Series(["C"]), pd.Series(["Z"]))
        assert result.iloc[0] == "MTO"

    def test_output_length_matches_input(self):
        abc = pd.Series(["A", "B", "C"] * 3)
        xyz = pd.Series(["X", "Y", "Z"] * 3)
        assert len(classify_mts_mto(abc, xyz)) == len(abc)

    def test_preserves_index(self):
        abc = pd.Series(["A", "B", "C"], index=[10, 20, 30])
        xyz = pd.Series(["X", "Y", "Z"], index=[10, 20, 30])
        result = classify_mts_mto(abc, xyz)
        assert list(result.index) == [10, 20, 30]
