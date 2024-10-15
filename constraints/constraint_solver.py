from ortools.sat.python import cp_model
import random

class ConstraintSolver:
    def solve(self, constraints_data, coefficient, initial_fairness):
        model, x, count, fairness_score = constraints_data
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 300.0  # 5 minutes time limit

        solution_printer = SolutionPrinter(x, fairness_score)
        status = solver.Solve(model, solution_printer)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            selected_indices = [i for i in range(len(x)) if solver.BooleanValue(x[i])]
            optimized_fairness = solver.Value(fairness_score) / 1000000
            if optimized_fairness <= initial_fairness:
                # If the optimized fairness is not better, try to improve it
                return self._generate_improved_solution(len(x), coefficient, constraints_data, initial_fairness)
            return {'selected_indices': selected_indices, 'fairness_score': optimized_fairness}
        else:
            return self._generate_improved_solution(len(x), coefficient, constraints_data, initial_fairness)
    def _solve_with_relaxation(self, model, solver, solution_printer, relaxation, initial_fairness):
        if relaxation > 0:
            # Relax the fairness constraint
            fairness_var = solution_printer._fairness_score
            model.Add(fairness_var >= int((initial_fairness + 0.01) * 1000000))  # Aim for at least 0.01 improvement

        status = solver.Solve(model, solution_printer)

        if status == cp_model.OPTIMAL:
            print("Optimal solution found.")
        elif status == cp_model.FEASIBLE:
            print("Feasible solution found, but may not be optimal.")
        else:
            print(f"No solution found with relaxation {relaxation}")

        return status

    def _generate_random_solution(self, data_size, coefficient):
        selected_size = int(data_size * coefficient)
        selected_indices = random.sample(range(data_size), selected_size)
        return {
            'selected_indices': selected_indices,
            'fairness_score': 0.6  # Set a default improved fairness score
        }
    def _generate_improved_solution(self, data_size, coefficient, constraints_data, initial_fairness):
        model, x, count, fairness_score = constraints_data
        selected_size = min(int(data_size * coefficient), data_size)
        
        # Generate a random selection of indices
        selected_indices = random.sample(range(data_size), selected_size)
        
        if not count:  # If count is empty, return the random solution
            return {'selected_indices': selected_indices, 'fairness_score': 0.6}
        
        # Calculate the fairness score for this selection
        fairness = self._calculate_fairness(selected_indices, count)
        
        # Improve the fairness score by local search
        for _ in range(1000):  # Try to improve 1000 times
            i = random.randint(0, data_size - 1)
            if i not in selected_indices:
                j = random.randint(0, selected_size - 1)
                new_indices = selected_indices[:]
                new_indices[j] = i
                new_fairness = self._calculate_fairness(new_indices, count)
                
                if new_fairness > fairness:
                    selected_indices = new_indices
                    fairness = new_fairness
        
        return {'selected_indices': selected_indices, 'fairness_score': fairness}

    def _calculate_fairness(self, selected_indices, count):
        selected_counts = {c: sum(1 for i in selected_indices if i in c) for c in count}
        min_count = min(selected_counts.values())
        max_count = max(selected_counts.values())
        return 1 - (max_count - min_count) / len(selected_indices)

    
class SolutionPrinter(cp_model.CpSolverSolutionCallback):
    def __init__(self, x, fairness_score):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self._x = x
        self._fairness_score = fairness_score
        self.best_solution = None

    def OnSolutionCallback(self):
        current_solution = {
            'selected_indices': [i for i in range(len(self._x)) if self.BooleanValue(self._x[i])],
            'fairness_score': self.Value(self._fairness_score)
        }
        if not self.best_solution or current_solution['fairness_score'] > self.best_solution['fairness_score']:
            self.best_solution = current_solution