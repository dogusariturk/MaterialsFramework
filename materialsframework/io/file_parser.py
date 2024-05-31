"""
This module provides a class to parse files and extract compositions of elements.
"""
import os

import pandas as pd
from pymatgen.core import Composition, Element

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class FileParser:
    """
    A class that represents a file parser.
    
    This class provides methods to parse files and extract compositions of elements.
    """

    def __init__(self) -> None:
        """
        Initializes the Parser object.
        """
        self._element_list: list[str, ...] = [Element.from_Z(i).symbol for i in range(1, 103)]

    def parse(self, filename: str) -> pd.DataFrame:
        """
        Parses the given file and extracts the compositions of elements.

        Returns:
            pandas.DataFrame: The DataFrame containing the extracted compositions.

        Raises:
            ValueError: If the file type is not supported.
        """
        file_type = os.path.splitext(filename)[1][1:]
        if file_type == "xlsx":
            dataframe = pd.read_excel(filename)
        elif file_type == "csv":
            dataframe = pd.read_csv(filename)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        return self._process_dataframe(dataframe)

    def _process_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Processes the given DataFrame by performing specific operations on it.

        Args:
            dataframe (pandas.DataFrame): The input DataFrame to be processed.

        Returns:
            pandas.DataFrame: The processed DataFrame.
        """
        new_dataframe = dataframe[dataframe.columns.intersection(self._element_list)].copy()
        new_dataframe["Composition"] = new_dataframe.apply(
                lambda x: Composition(x.to_dict()), axis=1
        )

        return new_dataframe
