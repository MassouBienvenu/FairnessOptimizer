import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

class DataAdjuster:
    def __init__(self, fairness_evaluator):
        self.fairness_evaluator = fairness_evaluator

    def adjust_data(self, data, solution, coefficient, sensitive_attributes, max_iterations=10):
        original_size = len(data)
        initial_fairness = self.fairness_evaluator.evaluate(data, sensitive_attributes)

        if coefficient <= 1:
            # Undersampling
            adjusted_data = data.iloc[solution['selected_indices']]
            lines_removed = original_size - len(adjusted_data)
            return adjusted_data, lines_removed

        else:
            # Oversampling
            target_size = int(original_size * coefficient)
            adjusted_data = data.copy()

            for _ in range(max_iterations):
                synthetic_samples = self._generate_synthetic_samples(data, target_size - len(adjusted_data))
                adjusted_data = pd.concat([adjusted_data, synthetic_samples], ignore_index=True)
                
                current_fairness = self.fairness_evaluator.evaluate(adjusted_data, sensitive_attributes)
                
                if current_fairness > initial_fairness:
                    break
                
                # If fairness didn't improve, remove the last batch of synthetic samples
                adjusted_data = adjusted_data.iloc[:len(adjusted_data) - len(synthetic_samples)]

            lines_added = len(adjusted_data) - original_size
            return adjusted_data, lines_added
    def _generate_synthetic_samples(self, data, n_samples):
        synthetic_samples = []
        sensitive_attributes = [col for col in data.columns if col.startswith('sensitive_')]
        
        for _ in range(n_samples):
            # Select a random combination of sensitive attributes
            combo = {attr: np.random.choice(data[attr].unique()) for attr in sensitive_attributes}
            
            # Find samples matching this combination
            matching_samples = data[(data[sensitive_attributes] == pd.Series(combo)).all(axis=1)]
            
            if len(matching_samples) > 0:
                # If there are matching samples, choose one randomly
                original_sample = matching_samples.sample(n=1).iloc[0]
            else:
                # If no matching samples, choose a random sample
                original_sample = data.sample(n=1).iloc[0]
            
            synthetic_sample = original_sample.copy()
            
            for column in data.columns:
                if column not in sensitive_attributes:
                    column_type = data[column].dtype
                    if np.issubdtype(column_type, np.integer):
                        # For integer columns, generate a random integer within the range
                        min_val, max_val = data[column].min(), data[column].max()
                        synthetic_sample[column] = np.random.randint(min_val, max_val + 1)
                    elif np.issubdtype(column_type, np.floating):
                        # For float columns, add small random perturbation
                        std_dev = data[column].std() * 0.1
                        synthetic_sample[column] += np.random.normal(0, std_dev)
                    elif column_type == 'object' or column_type == 'category':
                        # For categorical or string columns, randomly choose a value from the column
                        synthetic_sample[column] = np.random.choice(data[column].unique())
                    else:
                        # For other types, keep the original value
                        pass
            
            synthetic_samples.append(synthetic_sample)
        
        return pd.DataFrame(synthetic_samples)