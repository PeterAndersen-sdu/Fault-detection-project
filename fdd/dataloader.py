# fdd/dataloader.py

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.io import loadmat

from .dataset import TimeSeriesDataset


class DataLoader:
    """
    Loads Cranfield Multiphase Flow Facility .mat files and returns structured datasets.
    """

    DEFAULT_SENSOR_COLUMNS = [
        "Air_Delivery_P",
        "P_Bottom_Riser",
        "P_Top_Riser",
        "P_Top_Separator",
        "P_3Phase_Separator",
        "Diff_P_Riser",
        "Diff_P_VC404",
        "Air_In_Flow",
        "Water_In_Flow",
        "Flow_Top_Riser",
        "Level_Top_Sep",
        "Flow_Top_Sep_Out",
        "Density_Top_Riser",
        "Density_Top_Sep_Out",
        "Density_Water_In",
        "Temp_Top_Riser",
        "Temp_Top_Sep_Out",
        "Temp_Water_In",
        "Level_3Phase_Sep",
        "Pos_VC501",
        "Pos_VC302",
        "Pos_VC101",
        "Pump_Current_PO1",
    ]

    # Initializes with path to .mat file and optional sensor column names.
    def __init__(
        self,
        sensor_columns: Optional[List[str]] = None,
    ) -> None:
        self.sensor_columns = sensor_columns or self.DEFAULT_SENSOR_COLUMNS

    # Loads the .mat file and extracts datasets
    def load(self, file: str) -> Dict[str, TimeSeriesDataset]:
        """
        Returns a dictionary like:
        {
            "Set1_1": TimeSeriesDataset(...),
            "Set1_2": TimeSeriesDataset(...),
            "Set1_3": TimeSeriesDataset(...),
        }
        """
        if file == "FaultyCase1" or file == "fc1" or file == "1":
            file_path = "data/FaultyCase1.mat"
        elif file == "Training" or file == "training" or file == "train":
            file_path = "data/Training.mat"
        elif file == "FaultyCase2" or file == "fc2" or file == "2":
            file_path = "data/FaultyCase2.mat"
        elif file == "FaultyCase3" or file == "fc3" or file == "3":
            file_path = "data/FaultyCase3.mat"
        elif file == "FaultyCase4" or file == "fc4" or file == "4":
            file_path = "data/FaultyCase4.mat"
        elif file == "FaultyCase5" or file == "fc5" or file == "5":
            file_path = "data/FaultyCase5.mat"
        elif file == "FaultyCase6" or file == "fc6" or file == "6":
            file_path = "data/FaultyCase6.mat"
        else:
            raise ValueError(f"Unknown file identifier: {file}")
        
        raw = loadmat(file_path, squeeze_me=True, struct_as_record=False)
        
        set_names = [name for name in raw.keys() if name.startswith("Set") or name.startswith("T")]

        datasets = {}
        for set_name in set_names:
            sensors_df = self._extract_set(raw, set_name)

            datasets[set_name] = TimeSeriesDataset(
                name=set_name,
                sensors=sensors_df,
                metadata={"source_file": str(file_path), "set_name": set_name},
            )

        return datasets

    # Extracts sensor data for one set from the loaded .mat dictionary.
    def _extract_set(self, raw: dict, set_name: str) -> tuple[pd.DataFrame]:
        """
        Extract one set from the loaded .mat dictionary.

        NOTE:
        The exact indexing into your .mat file may need a small adjustment depending
        on the true MATLAB struct layout in your local file.
        """
        if set_name not in raw:
            raise KeyError(f"{set_name} not found in .mat file")

        set_obj = raw[set_name]

        # The Set* entries are already 2D numeric arrays in this dataset.
        sensor_matrix = np.asarray(set_obj[:, :23], dtype=float)

        sensors_df = pd.DataFrame(sensor_matrix, columns=self.sensor_columns)

        return sensors_df