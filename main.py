import os
import time
from fairness.fairness_evaluator import FairnessEvaluator
from pattern_mining.pattern_miner import PatternMiner
from pattern_mining.pattern_evaluator import PatternEvaluator
from constraints.constraint_generator import ConstraintGenerator
from constraints.constraint_solver import ConstraintSolver
from data.data_adjuster import DataAdjuster
from reporting.report_generator import ReportGenerator

class FairnessOptimizer:
    def __init__(self, config, data):
        self.config = config
        self.data = data
        self.fairness_evaluator = FairnessEvaluator()
        self.pattern_miner = PatternMiner()
        self.constraint_generator = ConstraintGenerator()
        self.constraint_solver = ConstraintSolver()
        self.data_adjuster = DataAdjuster(self.fairness_evaluator)
        self.report_generator = ReportGenerator()
        
        # Mine patterns first
        patterns = self.pattern_miner.fp_growth(self.data)
        
        # Initialize PatternEvaluator with the required arguments
        self.pattern_evaluator = PatternEvaluator(self.data, self.config['sensitive_attributes'], patterns)
        # Evaluate patterns for the original dataset
        self.original_patterns = self.pattern_evaluator.evaluate_pattern({frozenset(pattern): count for pattern, count in patterns.items()})
    
    
    def calculate_optimal_coefficients(self, initial_fairness, optimized_fairness_score):
        def binary_search(low, high):
            while high - low > 0.001:
                mid = (low + high) / 2
                adjusted_data, _ = self.data_adjuster.adjust_data(
                    self.data, 
                    {'selected_indices': range(len(self.data))}, 
                    mid, 
                    self.config['sensitive_attributes']
                )
                fairness = self.fairness_evaluator.evaluate(adjusted_data, self.config['sensitive_attributes'])
                if abs(fairness - optimized_fairness_score) < 0.001:
                    return mid
                elif fairness < optimized_fairness_score:
                    low = mid
                else:
                    high = mid
            return (low + high) / 2

        coeff_less_than_1 = binary_search(0, 1)
        coeff_greater_than_1 = binary_search(1, 2)  # Assuming a maximum of 2x oversampling

        return coeff_less_than_1, coeff_greater_than_1
    def optimize_fairness(self):
        print("Starting fairness optimization process...")
        start_time = time.time()

        try:
            # Step 1: Evaluate initial fairness
            initial_fairness = self.fairness_evaluator.evaluate(self.data, self.config['sensitive_attributes'])
            print(f"Initial fairness: {initial_fairness}")

            # Step 2: Generate constraints
            constraints_data = self.constraint_generator.generate_constraints(self.data, self.config)

            # Step 3: Solve constraints
            solution = self.constraint_solver.solve(constraints_data, self.config['coefficient'], initial_fairness)
            optimized_fairness_score = solution['fairness_score']
            print(f"Optimized fairness score: {optimized_fairness_score}")

            # Calculate optimal coefficients
            coeff_less_than_1, coeff_greater_than_1 = self.calculate_optimal_coefficients(initial_fairness, optimized_fairness_score)
            print(f"Optimal coefficient < 1: {coeff_less_than_1:.4f}")
            print(f"Optimal coefficient > 1: {coeff_greater_than_1:.4f}")

            # Step 4: Adjust data based on solution
            adjusted_data, _ = self.data_adjuster.adjust_data(self.data, solution, self.config['coefficient'], self.config['sensitive_attributes'])

            # Step 5: Evaluate final fairness
            final_fairness = self.fairness_evaluator.evaluate(adjusted_data, self.config['sensitive_attributes'])
            print(f"Final fairness: {final_fairness}")

            # Check if final fairness is less than initial fairness
            if final_fairness < initial_fairness:
                print("Warning: Final fairness is less than initial fairness. Reverting to original dataset.")
                adjusted_data = self.data
                final_fairness = initial_fairness

            # Step 6: Get patterns
            print("Step 6: Evaluating patterns...")
            patterns = self.pattern_evaluator.evaluate_pattern({frozenset(pattern): count for pattern, count in self.pattern_miner.fp_growth(adjusted_data).items()})
            print(f"Number of patterns found: {len(patterns)}")

            # Step 7: Generate report
            print("Step 7: Generating report...")
            report_path = self.report_generator.generate_report(
                initial_fairness, 
                final_fairness, 
                self.config, 
                self.data,  # Original data
                adjusted_data,  # Adjusted data
                optimized_fairness_score
            )
            print(f"Report generated at: {report_path}")

            # Step 8: Save optimized dataset
            print("Step 8: Saving optimized dataset...")
            dataset_path = os.path.join(os.getcwd(), 'optimized_dataset.csv')
            adjusted_data.to_csv(dataset_path, index=False)
            print(f"Optimized dataset saved at: {dataset_path}")

            end_time = time.time()
            print(f"Fairness optimization process completed in {end_time - start_time:.2f} seconds.")
            print(f"Initial fairness score: {initial_fairness:.4f}")
            print(f"Final fairness score: {final_fairness:.4f}")

            return dataset_path, report_path, adjusted_data

        except Exception as e:
            print(f"Error during fairness optimization: {str(e)}")
            raise