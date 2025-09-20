import os
import numpy as np
import re

def measure_accuracy(solutions_dir, matches_dir):
    match_counts = []

    for filename in sorted(os.listdir(solutions_dir), key=lambda x: int(re.search(r'(\d+)\.txt$', x).group(1))):
        if filename.endswith(".txt"):
            solution_path = os.path.join(solutions_dir, filename)
            match_path = os.path.join(matches_dir, filename)

            # Load files
            solution = np.loadtxt(solution_path, dtype=int)
            match_array = np.loadtxt(match_path, dtype=int)

            mask = solution != -1
            match_bool = solution[mask] == match_array[mask]
            n_matches = np.sum(match_bool)
            match_counts.append(n_matches/np.sum(mask))

    return np.mean(match_counts), np.std(match_counts), np.min(match_counts), np.max(match_counts)

if __name__ == "__main__":
    solutions_dir = "/home/daniel/pythonny/MPI/SeePelican/tests/full_worms/solutions"
    matches_dir = "/home/daniel/pythonny/MPI/SeePelican/tests/full_worms/matches"

    mean_accuracy, std_accuracy = measure_accuracy(solutions_dir, matches_dir)
    print(f"Mean accuracy: {mean_accuracy:.4f}")
    print(f"Standard deviation: {std_accuracy:.4f}")
