import glob
import re
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import pylibmgm

def graph_match_worm(dd_file_path, solution_file_path, idx):
    try:
        model = pylibmgm.io.parse_dd_file_gm(dd_file_path)
        solution = pylibmgm.solver.solve_gm(model)
        np.savetxt(solution_file_path, solution.labeling(), fmt="%d")
        return (idx, solution.evaluate())
    except Exception as e:
        print(f"Error processing {dd_file_path}: {e}")
        return None

def graph_match_all_worms(dd_file_directory, solution_file_directory, num_workers=12):
    dd_file_paths = sorted(glob.glob(dd_file_directory + "/*.dd"), key=lambda x: int(re.search(r'(\d+)\.dd$', x).group(1)))
    solution_file_paths = [f"{solution_file_directory}/worm_{i:03d}.txt" for i in range(len(dd_file_paths))]
    index = [i for i in range(len(dd_file_paths))]
    results = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(graph_match_worm, dd_path, sol_path, idx) for dd_path, sol_path, idx in zip(dd_file_paths, solution_file_paths, index)]

        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    ordered = np.zeros(len(index))
    for idx, cost in results:
        ordered[idx] = cost
    # print(ordered)
    
    return results

if __name__ == "__main__":
    graph_match_all_worms(
        "/home/daniel/pythonny/MPI/SeePelican/tests/full_worms/dd_files",
        "/home/daniel/pythonny/MPI/SeePelican/tests/full_worms/matches",
        num_workers=12
    )