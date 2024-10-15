import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Set
from itertools import combinations

class PatternEvaluator:
    def __init__(self, data: pd.DataFrame, sensitive_attributes: List[str], patterns: Dict[frozenset, int]):
        """
        Initialize the PatternEvaluator class.

        Args:
            data (pd.DataFrame): The dataset to evaluate patterns on.
            sensitive_attributes (List[str]): List of column names for sensitive attributes.
            patterns (Dict[frozenset, int]): Dictionary of patterns (itemsets) and their support counts.
        """
        self.data = data
        self.sensitive_attributes = sensitive_attributes
        self.patterns = patterns
        self.total_records = len(data)
        
    def evaluate_pattern_support(self, data: pd.DataFrame, pattern: frozenset) -> float:
        """
        Evaluate the support of a given pattern in the dataset.

        :param data: DataFrame containing the dataset
        :param pattern: frozenset representing the pattern to evaluate
        :return: Support value of the pattern (between 0 and 1)
        """
        # Create a boolean mask for each condition in the pattern
        masks = []
        for item in pattern:
            if item in data.values:  # Check if the item exists in the DataFrame values
                masks.append(data.isin([item]).any(axis=1))  # Create a mask for rows containing the item
            else:
                raise ValueError(f"Item '{item}' not found in the dataset")

        # Combine all masks using logical AND
        combined_mask = pd.Series(True, index=data.index)
        for mask in masks:
            combined_mask &= mask

        # Calculate the support
        support = combined_mask.mean()

        return support
    
    def calculate_pattern_distribution(self, pattern: frozenset, subgroup: pd.DataFrame = None) -> Dict[str, Dict[str, float]]:
        """
        Analyzes pattern distribution across subgroups for each sensitive attribute.

        Args:
            pattern (frozenset): The pattern to analyze.
            subgroup (pd.DataFrame): Optional subgroup to analyze. If None, uses the full data.

        Returns:
            Dict[str, Dict[str, float]]: A nested dictionary with sensitive attributes as outer keys,
            their values as inner keys, and the pattern support within each subgroup as values.
        """
        distribution = {}
        data_to_use = subgroup if subgroup is not None else self.data
        for attr in self.sensitive_attributes:
            distribution[attr] = {}
            for value in data_to_use[attr].unique():
                subgroup = data_to_use[data_to_use[attr] == value]
                support = sum(pattern.issubset(set(row)) for row in subgroup.itertuples(index=False, name=None))
                distribution[attr][value] = support / len(subgroup) if len(subgroup) > 0 else 0
        return distribution

   
    def intersection_fairness(self, pattern: frozenset, k: int = 2) -> Dict[Tuple[str, ...], float]:
        """
        Analyzes patterns across intersectional subgroups.

        Args:
            pattern (frozenset): The pattern to analyze.
            k (int): The number of sensitive attributes to consider in each intersection. Default is 2.

        Returns:
            Dict[Tuple[str, ...], float]: A dictionary with combinations of sensitive attributes as keys
            and their intersection fairness measure for the given pattern as values.
        """
        intersections = {}
        for attrs in combinations(self.sensitive_attributes, k):
            subgroups = self.data.groupby(list(attrs))
            supports = []
            for _, subgroup in subgroups:
                support = sum(pattern.issubset(set(row)) for row in subgroup.itertuples(index=False, name=None))
                supports.append(support / len(subgroup))
            intersections[attrs] = max(supports) - min(supports)
        return intersections


    def evaluate_pattern(self, pattern: Dict[frozenset, int]) -> Dict[frozenset, Dict[str, float]]:
        """
        Evaluates a single pattern using all fairness metrics.

        Args:
            pattern (Dict[frozenset, int]): The pattern to evaluate, with support count.

        Returns:
            Dict[frozenset, Dict[str, float]]: A nested dictionary containing all fairness metric results for the pattern.
        """
        results = {}
        for pattern_set, count in pattern.items():
            support = count / self.total_records
            results[pattern_set] = {
                "support": support,
                "intersection_fairness": self.intersection_fairness(pattern_set)
            }
        return results

    def evaluate_all_patterns(self) -> Dict[frozenset, Dict[str, Dict[str, float]]]:
        """
        Evaluates all patterns using all fairness metrics.

        Returns:
            Dict[frozenset, Dict[str, Dict[str, float]]]: A nested dictionary containing all fairness metric results 
            for all patterns.
        """
        results = {}
        for pattern in self.patterns:
            results[pattern] = self.evaluate_pattern(pattern)
        return results



    