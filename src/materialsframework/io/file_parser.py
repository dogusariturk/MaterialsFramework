"""Parses Excel/CSV files and extracts per-row elemental compositions.

`FileParser` matches spreadsheet columns against element symbols and builds a
`pymatgen.core.Composition` for each row.
"""

from pathlib import Path

import pandas as pd
from pymatgen.core import Composition, Element

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class FileParser:
    """Parses Excel and CSV files and extracts elemental compositions as pymatgen `Composition` objects."""

    def __init__(self) -> None:
        """Builds the internal list of element symbols for atomic numbers 1 through 102."""
        self._element_list: list[str] = [Element.from_Z(i).symbol for i in range(1, 103)]

    def parse(self, filename: str) -> pd.DataFrame:
        """Reads an Excel (.xlsx) or CSV (.csv) file and extracts elemental compositions from its columns.

        Args:
            filename (str): The path to the file to parse.

        Returns:
            pandas.DataFrame: A DataFrame containing the extracted elemental compositions.

        Raises:
            ValueError: If the file type is not supported.
        """
        file_type = Path(filename).suffix[1:]
        if file_type == "xlsx":
            dataframe = pd.read_excel(filename)
        elif file_type == "csv":
            dataframe = pd.read_csv(filename)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        return self._process_dataframe(dataframe)

    def _process_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Keeps only the columns matching `_element_list` and adds a "Composition" column built from them.

        Args:
            dataframe (pandas.DataFrame): The input DataFrame containing raw data.

        Returns:
            pandas.DataFrame: The processed DataFrame with an additional "Composition" column.
        """
        new_dataframe = dataframe[dataframe.columns.intersection(self._element_list)].copy()
        new_dataframe["Composition"] = new_dataframe.apply(lambda x: Composition(x.to_dict()), axis=1)

        return new_dataframe
