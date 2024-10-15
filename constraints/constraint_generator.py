
from ortools.sat.python import cp_model
from itertools import combinations as iter_combinations, product

import pandas as pd
class ConstraintGenerator:
    def __init__(self):
        self.test_set_size = 0

    def generate_constraints(self, data, config):
        model = cp_model.CpModel()
        
        n = len(data)
        x = [model.NewBoolVar(f'x[{i}]') for i in range(n)]
        
        k = int(n * config['coefficient'])
        model.Add(sum(x) == k)
        
        combinations = self._get_combinations(data, config['sensitive_attributes'])
        count = {}
        for c in combinations:
            count[c] = model.NewIntVar(0, k, f'count[{c}]')
            model.Add(count[c] == sum(x[i] for i in range(n) if self._has_combination(data.iloc[i], c)))
        
        fairness_score = model.NewIntVar(0, 1000000, 'fairness_score')
        min_count = model.NewIntVar(0, k, 'min_count')
        max_count = model.NewIntVar(0, k, 'max_count')
        
        model.AddMinEquality(min_count, [count[c] for c in count])
        model.AddMaxEquality(max_count, [count[c] for c in count])
        
        # Use multiplication instead of division
        model.Add(fairness_score * k == 1000000 * k - (max_count - min_count) * 1000000)
        
        model.Maximize(fairness_score)
        
        return model, x, count, fairness_score
    def _calculate_initial_fairness(self, data, sensitive_attributes):
        combinations = self._get_combinations(data, sensitive_attributes)
        supports = {}
        total_records = len(data)
        for combo in combinations:
            mask = pd.Series(True, index=data.index)
            for attr, value in combo:
                mask &= (data[attr] == value)
            support = mask.sum() / total_records
            supports[combo] = support
        return 1 - (max(supports.values()) - min(supports.values()))

    def get_all_constraints(self):
        constraints = []
        constraints.append(f"Test set size: {self.test_set_size}")
        return constraints

    # Keep the _calculate_initial_fairness, _get_combinations, and _has_combination methods as they are
    def _get_combinations(self, data, sensitive_attributes):
        all_combinations = []
        for attr in sensitive_attributes:
            values = data[attr].unique()
            all_combinations.extend([frozenset([(attr, value)]) for value in values])
        
        for k in range(2, len(sensitive_attributes) + 1):
            for attrs in iter_combinations(sensitive_attributes, k):
                for values in product(*[data[attr].unique() for attr in attrs]):
                    all_combinations.append(frozenset(zip(attrs, values)))
        
        return all_combinations

    def _has_combination(self, row, combination):
        return all(row[attr] == value for attr, value in combination)


