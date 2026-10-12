
import pandas as pd
import pytest

from app.feature_engineering import engineer_features


def test_engineered_features_are_calculated():
    data = pd.DataFrame({
        "Tenure": [10],
        "Total Spend": [500.0],
        "Support Calls": [4],
    })

    result = engineer_features(data)

    assert result.loc[0, "AverageSpendPerTenure"] == 50.0
    assert result.loc[0, "SupportCallsPerTenure"] == 0.4


def test_original_data_is_not_modified():
    data = pd.DataFrame({
        "Tenure": [10],
        "Total Spend": [500.0],
        "Support Calls": [4],
    })

    original_columns = list(data.columns)

    engineer_features(data)

    assert list(data.columns) == original_columns


def test_zero_tenure_is_rejected():
    data = pd.DataFrame({
        "Tenure": [0],
        "Total Spend": [500.0],
        "Support Calls": [4],
    })

    with pytest.raises(ValueError, match="Tenure must be greater than zero"):
        engineer_features(data)
