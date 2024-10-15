import pandas as pd
import numpy as np
from typing import List, Dict
from itertools import combinations as iter_combinations, product

class FairnessEvaluator:
    def __init__(self):
        self.combinations_cache = {}

    def evaluate(self, data: pd.DataFrame, sensitive_attributes: List[str]) -> float:
        combinations = self._get_combinations(data, sensitive_attributes)
        supports = self._calculate_supports(data, combinations)
        fairness_score = 1 - (max(supports.values()) - min(supports.values()))
        return fairness_score

    def _get_combinations(self, data, sensitive_attributes):
        key = tuple(sensitive_attributes)
        if key not in self.combinations_cache:
            all_combinations = []
            for attr in sensitive_attributes:
                values = data[attr].unique()
                all_combinations.extend([frozenset([(attr, value)]) for value in values])
            
            for k in range(2, len(sensitive_attributes) + 1):
                for attrs in iter_combinations(sensitive_attributes, k):
                    for values in product(*[data[attr].unique() for attr in attrs]):
                        all_combinations.append(frozenset(zip(attrs, values)))
            
            self.combinations_cache[key] = all_combinations
        
        return self.combinations_cache[key]

    def _calculate_supports(self, data: pd.DataFrame, combinations: List[frozenset]) -> Dict[frozenset, float]:
        supports = {}
        total_records = len(data)

        for combo in combinations:
            mask = pd.Series(True, index=data.index)
            for attr, value in combo:
                mask &= (data[attr] == value)
            support = mask.sum() / total_records
            supports[combo] = support

        return supports