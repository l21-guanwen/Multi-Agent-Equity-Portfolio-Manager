"""
CSV loading utilities.

Provides standardized CSV loading with type conversion and validation.
"""

from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional, Type, TypeVar

import pandas as pd
from pydantic import BaseModel

from app.core.config import get_settings
from app.core.constants import DataFileName

T = TypeVar("T", bound=BaseModel)


class CSVLoader:
    """
    Utility class for loading CSV data files.
    
    Handles:
    - File path resolution
    - Type conversion (dates, floats, bools)
    - Missing value handling
    - Conversion to Pydantic models
    
    Example:
        loader = CSVLoader()
        df = loader.load_dataframe(DataFileName.BENCHMARK)
        models = loader.load_as_models(DataFileName.ALPHA, AlphaScore)
    """

    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize CSV loader.
        
        Args:
            data_path: Path to data directory. Uses settings if not provided.
        """
        if data_path is None:
            settings = get_settings()
            data_path = settings.csv_data_path
        self._data_path = Path(data_path)

    @property
    def data_path(self) -> Path:
        """Get the data directory path."""
        return self._data_path

    def get_file_path(self, filename: str | DataFileName) -> Path:
        """
        Get full path to a data file.
        
        Args:
            filename: Filename or DataFileName enum
            
        Returns:
            Full path to the file
        """
        if isinstance(filename, DataFileName):
            filename = filename.value
        return self._data_path / filename

    def file_exists(self, filename: str | DataFileName) -> bool:
        """Check if a data file exists."""
        return self.get_file_path(filename).exists()

    def load_dataframe(
        self,
        filename: str | DataFileName,
        parse_dates: Optional[list[str]] = None,
        date_columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Load a CSV file as a pandas DataFrame.
        
        Args:
            filename: Filename or DataFileName enum
            parse_dates: Columns to parse as dates
            date_columns: Alternative name for parse_dates
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        file_path = self.get_file_path(filename)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Combine date column parameters
        dates_to_parse = parse_dates or date_columns or []
        
        # Default date columns if not specified
        if not dates_to_parse:
            dates_to_parse = ["As_Of_Date", "as_of_date"]
        
        df = pd.read_csv(
            file_path,
            parse_dates=[c for c in dates_to_parse if c],
        )
        
        # Normalize column names (strip whitespace)
        df.columns = df.columns.str.strip()
        
        return df

    def load_as_models(
        self,
        filename: str | DataFileName,
        model_class: Type[T],
        column_mapping: Optional[dict[str, str]] = None,
    ) -> list[T]:
        """
        Load CSV and convert rows to Pydantic models.
        
        Args:
            filename: Filename or DataFileName enum
            model_class: Pydantic model class to convert to
            column_mapping: Optional mapping from CSV columns to model fields
            
        Returns:
            List of model instances
        """
        df = self.load_dataframe(filename)
        
        # Apply column mapping if provided
        if column_mapping:
            df = df.rename(columns=column_mapping)
        
        # Convert column names to snake_case for model compatibility
        df.columns = [self._to_snake_case(c) for c in df.columns]
        
        # Convert DataFrame to list of dicts and then to models
        records = df.to_dict(orient="records")
        models = []
        
        for record in records:
            # Clean up record
            cleaned = self._clean_record(record)
            try:
                model = model_class.model_validate(cleaned)
                models.append(model)
            except Exception as e:
                # Log warning but continue
                print(f"Warning: Failed to parse record: {e}")
                continue
        
        return models

    def load_as_dict(
        self,
        filename: str | DataFileName,
        key_column: str,
        value_column: str,
    ) -> dict[str, Any]:
        """
        Load CSV as a dictionary mapping.
        
        Args:
            filename: Filename or DataFileName enum
            key_column: Column to use as dictionary keys
            value_column: Column to use as dictionary values
            
        Returns:
            Dictionary mapping key_column -> value_column
        """
        df = self.load_dataframe(filename)
        return dict(zip(df[key_column], df[value_column]))

    def load_matrix(
        self,
        filename: str | DataFileName,
        index_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load CSV as a matrix (for covariance matrix, etc.).
        
        Args:
            filename: Filename or DataFileName enum
            index_column: Column to use as row index
            
        Returns:
            DataFrame with proper indexing for matrix operations
        """
        df = self.load_dataframe(filename)
        
        if index_column and index_column in df.columns:
            df = df.set_index(index_column)
        
        # Remove non-numeric columns for matrix operations
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        return df[numeric_cols]

    def _to_snake_case(self, name: str) -> str:
        """Convert column name to snake_case."""
        # Handle common patterns
        result = name.strip()
        
        # Replace spaces and hyphens with underscores
        result = result.replace(" ", "_").replace("-", "_")
        
        # Convert CamelCase to snake_case
        import re
        result = re.sub(r"(?<!^)(?=[A-Z])", "_", result)
        
        return result.lower()

    def _clean_record(self, record: dict[str, Any]) -> dict[str, Any]:
        """
        Clean a record for model validation.
        
        Handles:
        - NaN values -> None
        - Date conversion
        - Boolean conversion
        """
        cleaned = {}
        
        for key, value in record.items():
            # Handle NaN values
            if pd.isna(value):
                cleaned[key] = None
                continue
            
            # Handle pandas Timestamp
            if isinstance(value, pd.Timestamp):
                cleaned[key] = value.date()
                continue
            
            # Handle datetime
            if isinstance(value, datetime):
                cleaned[key] = value.date()
                continue
            
            # Handle boolean strings
            if isinstance(value, str) and value.lower() in ("true", "false"):
                cleaned[key] = value.lower() == "true"
                continue
            
            cleaned[key] = value
        
        return cleaned

    def get_available_files(self) -> list[str]:
        """Get list of available CSV files in data directory."""
        if not self._data_path.exists():
            return []
        return [f.name for f in self._data_path.glob("*.csv")]

    def get_file_info(self, filename: str | DataFileName) -> dict[str, Any]:
        """
        Get information about a data file.
        
        Args:
            filename: Filename or DataFileName enum
            
        Returns:
            Dictionary with file info (path, exists, rows, columns)
        """
        file_path = self.get_file_path(filename)
        
        info = {
            "path": str(file_path),
            "exists": file_path.exists(),
            "rows": 0,
            "columns": [],
        }
        
        if info["exists"]:
            df = self.load_dataframe(filename)
            info["rows"] = len(df)
            info["columns"] = list(df.columns)
        
        return info

