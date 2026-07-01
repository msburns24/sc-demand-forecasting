import numpy as np
import pandas as pd
import pytest

from src.importance import importance_by_group, mean_abs_importance, sample_rows


class TestSampleRows:
    def test_returns_all_rows_when_n_exceeds_length(self):
        df = pd.DataFrame({"x": range(5)})
        out = sample_rows(df, n=10)
        assert len(out) == 5

    def test_caps_at_n_when_n_below_length(self):
        df = pd.DataFrame({"x": range(100)})
        out = sample_rows(df, n=10)
        assert len(out) == 10

    def test_deterministic_with_same_random_state(self):
        df = pd.DataFrame({"x": range(100)})
        a = sample_rows(df, n=10, random_state=7)
        b = sample_rows(df, n=10, random_state=7)
        assert a["x"].tolist() == b["x"].tolist()

    def test_index_is_reset(self):
        df = pd.DataFrame({"x": range(5)})
        out = sample_rows(df, n=3, random_state=1)
        assert out.index.tolist() == list(range(len(out)))


class TestMeanAbsImportance:
    def test_ranking_and_values(self):
        # feature 'b' has the largest mean |shap|, then 'a', then 'c'
        shap_values = np.array(
            [
                [1.0, -3.0, 0.1],
                [-1.0, 3.0, -0.1],
                [2.0, 4.0, 0.2],
            ]
        )
        out = mean_abs_importance(shap_values, ["a", "b", "c"]).set_index("feature")
        assert out.loc["a", "mean_abs_shap"] == pytest.approx(np.mean([1, 1, 2]), abs=1e-4)
        assert out.loc["b", "mean_abs_shap"] == pytest.approx(np.mean([3, 3, 4]), abs=1e-4)
        assert out.loc["c", "mean_abs_shap"] == pytest.approx(np.mean([0.1, 0.1, 0.2]), abs=1e-4)

    def test_sorted_descending_with_rank_column(self):
        shap_values = np.array([[1.0, 5.0, 2.0]])
        out = mean_abs_importance(shap_values, ["a", "b", "c"])
        assert out["feature"].tolist() == ["b", "c", "a"]
        assert out["rank"].tolist() == [1, 2, 3]

    def test_one_row_per_feature(self):
        shap_values = np.zeros((4, 3))
        out = mean_abs_importance(shap_values, ["a", "b", "c"])
        assert len(out) == 3


class TestImportanceByGroup:
    def test_per_group_ranking(self):
        # rows 0-1 -> group G1, rows 2-3 -> group G2
        shap_values = np.array(
            [
                [1.0, 0.5],
                [1.0, 0.5],
                [0.2, 3.0],
                [0.2, 3.0],
            ]
        )
        group = ["G1", "G1", "G2", "G2"]
        out = importance_by_group(shap_values, ["a", "b"], group)

        g1 = out[out["group"] == "G1"].set_index("feature")
        assert g1.loc["a", "rank_within_group"] == 1
        assert g1.loc["b", "rank_within_group"] == 2

        g2 = out[out["group"] == "G2"].set_index("feature")
        assert g2.loc["b", "rank_within_group"] == 1
        assert g2.loc["a", "rank_within_group"] == 2

    def test_one_row_per_group_feature_pair(self):
        shap_values = np.ones((4, 2))
        group = ["G1", "G1", "G2", "G2"]
        out = importance_by_group(shap_values, ["a", "b"], group)
        assert len(out) == 4  # 2 groups x 2 features

    def test_length_mismatch_raises(self):
        shap_values = np.ones((4, 2))
        with pytest.raises(ValueError):
            importance_by_group(shap_values, ["a", "b"], ["G1", "G1"])
