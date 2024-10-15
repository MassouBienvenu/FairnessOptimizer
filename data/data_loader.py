import pandas as pd
from typing import Dict, List

class DataLoader:
    """
    Responsible for loading and preprocessing the dataset.
    """

    def load_csv(self, file_path: str) -> pd.DataFrame:
        """
        Load a CSV file and return it as a pandas DataFrame.

        :param file_path: Path to the CSV file.
        :return: Pandas DataFrame containing the dataset.
        """
        try:
            return pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            raise
        except pd.errors.ParserError as e:
            print(f"Error: Failed to parse the CSV file. {e}")
            raise

    def encode_categorical(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables using one-hot encoding.

        :param data: Pandas DataFrame containing the dataset.
        :return: Pandas DataFrame with categorical variables encoded.
        """
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns
        return pd.get_dummies(data, columns=categorical_columns)

    def identify_sensitive_attributes(self, data: pd.DataFrame, sensitive_attrs: List[str]) -> pd.DataFrame:
        """
        Mark specified columns as sensitive attributes.

        :param data: Pandas DataFrame containing the dataset.
        :param sensitive_attrs: List of column names to be marked as sensitive.
        :return: Pandas DataFrame with sensitive attributes identified.
        """
        data_copy = data.copy()
        for attr in sensitive_attrs:
            if attr not in data.columns:
                print(f"Warning: Sensitive attribute '{attr}' not found in the dataset.")
            else:
                data_copy[attr] = f"sensitive_{attr}"
        return data_copy

    def load_and_preprocess(self, file_path: str, sensitive_attrs: List[str]) -> pd.DataFrame:
        """
        Load the dataset, encode categorical variables, and identify sensitive attributes.

        :param file_path: Path to the CSV file.
        :param sensitive_attrs: List of column names to be marked as sensitive.
        :return: Pandas DataFrame with the dataset preprocessed.
        """
        data = self.load_csv(file_path)
        data = self.encode_categorical(data)
        data = self.identify_sensitive_attributes(data, sensitive_attrs)
        return data