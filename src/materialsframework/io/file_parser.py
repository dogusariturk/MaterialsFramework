"""
This module provides a class for parsing files and extracting compositions of chemical elements.

The primary class, `FileParser`, supports reading data from various file formats
and processing the data to extract compositions of elements based on their presence in the file.
"""
import os

import pandas as pd
from pymatgen.core import Composition, Element

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class FileParser:
    """
    A class for parsing files and extracting compositions of chemical elements.

    The `FileParser` class supports parsing Excel and CSV files to extract
    elemental compositions, which are represented using the `Composition` class
    from Pymatgen.
    """
    def __init__(self) -> None:
        """
        Initializes the FileParser object and prepares a list of element symbols.

        This method sets up the internal list of element symbols (_element_list) corresponding to elements
        with atomic numbers from 1 to 102, which will be used in the parsing process.
        """
        self._element_list: list[str] = [Element.from_Z(i).symbol for i in range(1, 103)]

    def parse(self, filename: str) -> pd.DataFrame:
        """
        Parses the given file and extracts the compositions of elements.

        This method reads data from the specified file, which can be either an Excel (.xlsx) or
        CSV (.csv) file. It processes the data to extract compositions of chemical elements
        based on the presence of element symbols in the columns.

        Args:
            filename (str): The path to the file to be parsed.

        Returns:
            pandas.DataFrame: A DataFrame containing the extracted compositions of elements.

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
        Processes the given DataFrame by filtering it to include only columns corresponding to element symbols.

        This method filters the input DataFrame to retain only those columns that match
        the element symbols in `_element_list`. It then creates a new column "Composition"
        that contains the elemental composition for each row, represented as a `Composition` object.

        Args:
            dataframe (pandas.DataFrame): The input DataFrame containing raw data.

        Returns:
            pandas.DataFrame: The processed DataFrame with an additional "Composition" column.
        """
        new_dataframe = dataframe[dataframe.columns.intersection(self._element_list)].copy()
        new_dataframe["Composition"] = new_dataframe.apply(
                lambda x: Composition(x.to_dict()), axis=1
        )

        return new_dataframe
