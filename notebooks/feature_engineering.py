
import pandas as pd

def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    if (data["Tenure"] <= 0).any():
        raise ValueError("Tenure must be greater than zero.")

    data["AverageSpendPerTenure"] = (
        data["Total Spend"] / data["Tenure"]
    )

    data["SupportCallsPerTenure"] = (
        data["Support Calls"] / data["Tenure"]
    )

    return data
