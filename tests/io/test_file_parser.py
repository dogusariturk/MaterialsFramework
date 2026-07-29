"""Tests for io/file_parser.py, no external dependencies beyond pandas/pymatgen."""

from __future__ import annotations

import pandas as pd
import pytest
from pymatgen.core import Composition

from materialsframework.io.file_parser import FileParser


@pytest.fixture()
def parser() -> FileParser:
    """FileParser instance shared across tests in this module."""
    return FileParser()


def test_element_list_starts_with_hydrogen(parser: FileParser) -> None:
    """Element list starts with H (Z=1)."""
    assert parser._element_list[0] == "H"


def test_element_list_ends_with_no102(parser: FileParser) -> None:
    """Element list ends with No (Z=102)."""
    assert parser._element_list[-1] == "No"


def test_element_list_length(parser: FileParser) -> None:
    """Element list contains exactly 102 symbols."""
    assert len(parser._element_list) == 102


def test_process_dataframe_adds_composition_column(parser: FileParser) -> None:
    """_process_dataframe produces a 'Composition' column."""
    df = pd.DataFrame({"Fe": [0.5], "Ni": [0.5], "Temperature": [300]})
    result = parser._process_dataframe(df)
    assert "Composition" in result.columns


def test_process_dataframe_drops_non_element_columns(parser: FileParser) -> None:
    """Non-element columns are dropped from the output."""
    df = pd.DataFrame({"Fe": [0.5], "Ni": [0.5], "Temperature": [300]})
    result = parser._process_dataframe(df)
    assert "Temperature" not in result.columns


def test_process_dataframe_retains_element_columns(parser: FileParser) -> None:
    """Element columns are retained alongside the Composition column."""
    df = pd.DataFrame({"Fe": [0.5], "Ni": [0.5]})
    result = parser._process_dataframe(df)
    assert "Fe" in result.columns
    assert "Ni" in result.columns


def test_process_dataframe_composition_type(parser: FileParser) -> None:
    """Composition column contains pymatgen Composition objects."""
    df = pd.DataFrame({"Fe": [0.5], "Ni": [0.5]})
    result = parser._process_dataframe(df)
    assert isinstance(result["Composition"].iloc[0], Composition)


def test_process_dataframe_composition_values(parser: FileParser) -> None:
    """Composition object reflects the element fractions from the row."""
    df = pd.DataFrame({"Fe": [1.0], "Ni": [0.0]})
    result = parser._process_dataframe(df)
    comp = result["Composition"].iloc[0]
    assert comp["Fe"] == pytest.approx(1.0)
    assert comp["Ni"] == pytest.approx(0.0)


def test_process_dataframe_multiple_rows(parser: FileParser) -> None:
    """Each row gets its own Composition object."""
    df = pd.DataFrame({"Fe": [1.0, 0.0], "Ni": [0.0, 1.0]})
    result = parser._process_dataframe(df)
    assert len(result) == 2
    assert result["Composition"].iloc[0]["Fe"] == pytest.approx(1.0)
    assert result["Composition"].iloc[1]["Ni"] == pytest.approx(1.0)


def test_process_dataframe_no_element_columns_gives_only_composition(parser: FileParser) -> None:
    """DataFrame with no element columns produces a single Composition column."""
    df = pd.DataFrame({"Temperature": [300], "Pressure": [1]})
    result = parser._process_dataframe(df)
    assert list(result.columns) == ["Composition"]
    assert len(result) == 1


def test_parse_csv(parser: FileParser, tmp_path) -> None:
    """parse() reads a CSV file and returns correct compositions."""
    csv_file = tmp_path / "alloys.csv"
    csv_file.write_text("Fe,Ni,Temperature\n0.75,0.25,300\n")
    result = parser.parse(str(csv_file))
    assert "Composition" in result.columns
    assert result["Composition"].iloc[0]["Fe"] == pytest.approx(0.75)


def test_parse_xlsx(parser: FileParser, tmp_path) -> None:
    """parse() reads an Excel file and returns correct compositions."""
    pytest.importorskip("openpyxl")
    xlsx_file = tmp_path / "alloys.xlsx"
    df = pd.DataFrame({"Fe": [0.6], "Ni": [0.4], "Temperature": [500]})
    df.to_excel(xlsx_file, index=False)
    result = parser.parse(str(xlsx_file))
    assert "Composition" in result.columns
    assert result["Composition"].iloc[0]["Fe"] == pytest.approx(0.6)


def test_parse_unsupported_extension_raises(parser: FileParser, tmp_path) -> None:
    """parse() raises ValueError for unsupported file extensions."""
    bad_file = tmp_path / "data.json"
    bad_file.write_text("{}")
    with pytest.raises(ValueError, match="Unsupported file type"):
        parser.parse(str(bad_file))
