# fdd/dataset.py

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np


@dataclass
class TimeSeriesDataset:
    """
    Container for one Cranfield dataset split, e.g. Set1_1, Set1_2, or Set1_3.
    """
    name: str
    sensors: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Validates the dataset inputs when the object is created.
    def __post_init__(self) -> None:
        if not isinstance(self.sensors, pd.DataFrame):
            raise TypeError("sensors must be a pandas DataFrame")

    @property
    def n_samples(self) -> int:
        # Returns the number of rows in the sensor data.
        return len(self.sensors)

    @property
    def n_features(self) -> int:
        # Returns the number of sensor columns.
        return self.sensors.shape[1]

    @property
    def feature_names(self) -> List[str]:
        # Returns the sensor column names as a list.
        return list(self.sensors.columns)

    # Converts the sensor dataframe to a NumPy array.
    def to_numpy(self) -> np.ndarray:
        return self.sensors.to_numpy()

    # Creates a new dataset with only the selected row range.
    def subset_rows(self, start: int, end: int) -> "TimeSeriesDataset":
        return TimeSeriesDataset(
            name=f"{self.name}[{start}:{end}]",
            sensors=self.sensors.iloc[start:end].reset_index(drop=True),
            metadata=self.metadata.copy(),
        )

    # Returns a deep copy of the dataset and its metadata.
    def copy(self) -> "TimeSeriesDataset":
        return TimeSeriesDataset(
            name=self.name,
            sensors=self.sensors.copy(),
            metadata=self.metadata.copy(),
        )

    # Adds extra key-value metadata to the dataset.
    def add_metadata(self, **kwargs: Any) -> None:
        self.metadata.update(kwargs)

    # Builds a short readable summary of the dataset.
    def summary(self) -> str:
        return (
            f"TimeSeriesDataset(name={self.name}, "
            f"n_samples={self.n_samples}, "
            f"n_features={self.n_features})"
        )
