import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Union

def setup_logging(log_level: str = 'INFO') -> logging.Logger:
    """
    Set up logging for the application.

    Args:
        log_level (str): The logging level. Default is 'INFO'.

    Returns:
        logging.Logger: Configured logger object.
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def validate_input(data: pd.DataFrame, sensitive_attributes: List[str], target: str) -> None:
    """
    Validate input data and attributes.

    Args:
        data (pd.DataFrame): Input dataset.
        sensitive_attributes (List[str]): List of sensitive attribute column names.
        target (str): Target variable column name.

    Raises:
        ValueError: If input validation fails.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")
    
    if len(data) < 100:
        raise ValueError("Dataset must have at least 100 rows.")
    
    missing_columns = set(sensitive_attributes + [target]) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Columns not found in dataset: {missing_columns}")
    
    if data[target].isnull().any():
        raise ValueError("Target variable contains null values.")

def calculate_group_statistics(data: pd.DataFrame, attribute: str, target: str) -> Dict[str, Dict[str, float]]:
    """
    Calculate statistics for different groups in the data.

    Args:
        data (pd.DataFrame): Input dataset.
        attribute (str): The attribute to group by.
        target (str): The target variable.

    Returns:
        Dict[str, Dict[str, float]]: A dictionary of group statistics.
    """
    groups = data.groupby(attribute)
    stats = {}
    
    for name, group in groups:
        stats[name] = {
            'count': len(group),
            'mean': group[target].mean(),
            'std': group[target].std()
        }
    
    return stats

def encode_categorical(data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Encode categorical variables using one-hot encoding.

    Args:
        data (pd.DataFrame): Input dataset.
        columns (List[str]): List of column names to encode.

    Returns:
        pd.DataFrame: Dataset with encoded categorical variables.
    """
    return pd.get_dummies(data, columns=columns, drop_first=True)

def normalize_numerical(data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Normalize numerical columns to have zero mean and unit variance.

    Args:
        data (pd.DataFrame): Input dataset.
        columns (List[str]): List of numerical column names to normalize.

    Returns:
        pd.DataFrame: Dataset with normalized numerical columns.
    """
    for col in columns:
        data[col] = (data[col] - data[col].mean()) / data[col].std()
    return data

def calculate_correlation(data: pd.DataFrame, attribute1: str, attribute2: str) -> float:
    """
    Calculate the correlation between two attributes.

    Args:
        data (pd.DataFrame): Input dataset.
        attribute1 (str): Name of the first attribute.
        attribute2 (str): Name of the second attribute.

    Returns:
        float: Correlation coefficient between the two attributes.
    """
    return data[attribute1].corr(data[attribute2])

def detect_outliers(data: pd.DataFrame, column: str, method: str = 'IQR') -> pd.Series:
    """
    Detect outliers in a specific column of the dataset.

    Args:
        data (pd.DataFrame): Input dataset.
        column (str): Name of the column to check for outliers.
        method (str): Method to use for outlier detection. Either 'IQR' or 'zscore'.

    Returns:
        pd.Series: Boolean series indicating outlier status for each row.
    """
    if method == 'IQR':
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (data[column] < lower_bound) | (data[column] > upper_bound)
    elif method == 'zscore':
        z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
        return z_scores > 3
    else:
        raise ValueError("Invalid method. Choose either 'IQR' or 'zscore'.")

def generate_summary_statistics(data: pd.DataFrame) -> Dict[str, Dict[str, Union[float, int]]]:
    """
    Generate summary statistics for all columns in the dataset.

    Args:
        data (pd.DataFrame): Input dataset.

    Returns:
        Dict[str, Dict[str, Union[float, int]]]: Dictionary of summary statistics for each column.
    """
    summary = {}
    for column in data.columns:
        if data[column].dtype in ['int64', 'float64']:
            summary[column] = {
                'mean': data[column].mean(),
                'median': data[column].median(),
                'std': data[column].std(),
                'min': data[column].min(),
                'max': data[column].max()
            }
        elif data[column].dtype == 'object':
            summary[column] = {
                'unique_values': data[column].nunique(),
                'most_common': data[column].mode().iloc[0],
                'most_common_count': data[column].value_counts().iloc[0]
            }
    return summary