import numpy as np
import glob
import re
from hyperparams import HyperParams
from load_worms import load_worm_data, align_all_worms_with_transforms, generalized_procrustes, align_worms_to_standard_axes
from atlas import build_atlas
from build_dd_file import build_dd_file_all_worms
from graph_match_worms import graph_match_all_worms
from measure_accuracy import measure_accuracy
import time
import pickle
import argparse
import os

def build_atlas_and_data(atlas_data_dir, target_data_dir, split=np.arange(0, 10)):
    # Need to fix this if else statement to use new aligned test worm data. 
    if not (target_data_dir is None):
        # train_file_paths = sorted(glob.glob(f"{atlas_data_dir}/*.txt"), key=lambda x: int(re.search(r'(\d+)\.txt$', x).group(1)))
        # unaligned_pos_array, rad_array = load_worm_data(train_file_paths)
        with open("/fs/home/smola/code/CElegans/graph_matching/train_worms.pkl", "rb") as f:
            unaligned_pos_array = pickle.load(f)
        rad_array = np.zeros((unaligned_pos_array.shape[0], unaligned_pos_array.shape[1], 3))
        pos_array = generalized_procrustes(unaligned_pos_array)

        with open("/fs/home/smola/code/CElegans/graph_matching/test_worms.pkl", "rb") as f:
            unaligned_pos_target_array = pickle.load(f)
        rad_target_array = np.zeros((unaligned_pos_target_array.shape[0], unaligned_pos_target_array.shape[1], 3))
        
        pos_target_array = align_all_worms_with_transforms(unaligned_pos_target_array, np.mean(pos_array, axis=0))
    else:
        train_file_paths = sorted(glob.glob(f"{atlas_data_dir}/*.txt"), key=lambda x: int(re.search(r'(\d+)\.txt$', x).group(1)))
        full_unaligned_pos_array, full_rad_array = load_worm_data(train_file_paths)
        all_worms = np.arange(full_unaligned_pos_array.shape[0])
        train_mask = ~np.isin(all_worms, split)
        pos_array = generalized_procrustes(full_unaligned_pos_array[train_mask])
        
        rad_array = full_rad_array[train_mask]
        
        pos_target_array = align_all_worms_with_transforms(full_unaligned_pos_array[~train_mask], np.mean(pos_array, axis=0))
        rad_target_array = full_rad_array[~train_mask]
        
    atlas = build_atlas(pos_array, rad_array)
    return atlas, pos_target_array, rad_target_array

def process_worms(atlas, pos_target_array, rad_target_array, dd_file_dir, solution_file_dir, matches_file_dir, key_file_dir, hyperparams, subgraph_size=558, num_dd_file_workers=8, num_graph_match_workers=12):
    build_dd_file_all_worms(atlas, pos_target_array, rad_target_array, dd_file_dir, solution_file_dir, key_file_dir, hyperparams, subgraph_size=subgraph_size, num_workers=num_dd_file_workers)
    graph_match_all_worms(dd_file_dir, matches_file_dir, num_workers=num_graph_match_workers)
    return measure_accuracy(solution_file_dir, matches_file_dir)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process worms with configurable subgraph size')
    parser.add_argument('--subgraph_size', type=int, default=558, help='Subgraph size for processing (default: 558)')
    args = parser.parse_args()
    
    subgraph_size = args.subgraph_size
    
    start_time = time.time()
    atlas_data_dir = "/home/daniel/pythonny/MPI/SeePelican/data/worms/train/aligned"
    target_data_dir = "/home/daniel/pythonny/MPI/SeePelican/data/worms/test_set_2/aligned"
    
    # Update directory paths based on subgraph_size
    base_file_dir = f"/fs/gpfs41/lv11/fileset01/pool/pool-smola/full_worms_{subgraph_size}"
    dd_file_dir = f"{base_file_dir}/dd_files"
    solution_file_dir = f"{base_file_dir}/solutions"
    matches_file_dir = f"{base_file_dir}/matches"
    key_file_dir = f"{base_file_dir}/keys"
    atlas_pickle_dir = f"{base_file_dir}/atlas_pickle"
    
    # Create directories if they don't exist
    directories = [base_file_dir, dd_file_dir, solution_file_dir, matches_file_dir, key_file_dir, atlas_pickle_dir]
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Directory ready: {dir_path}")

    # script_path="/home/daniel/pythonny/MPI/SeePelican/run_graph_matching.sh"

    # hyperparams = HyperParams(c0=5000, dcen_threshold=20.0, lambda_off=1.0, lambda_cen=1.0039856553767617, lambda_rad=0.6210901242983923)
    # hyperparams = HyperParams(c0=5000, dcen_threshold=20.0, lambda_off=1.0, lambda_cen=1.0039856553767617, lambda_rad=0.6210901242983923)
    # hyperparams = HyperParams(c0=5000, dcen_threshold=20.0, lambda_off=1.0, lambda_cen=0.0, lambda_rad=0.0)
    # hyperparams = HyperParams(c0=3000, dcen_threshold=20.0, lambda_off=0.811, lambda_cen=0.479, lambda_rad=0.336)
    
    # hyperparams = HyperParams(c0=10000, dcen_threshold=2000000.0, lambda_off=1.0, lambda_cen=2.0079713107535233, lambda_rad=1.2421802485967846)
    hyperparams = HyperParams(c0=10000, dcen_threshold=20.0, lambda_off=1.0, lambda_cen=0.0023008084069785772, lambda_rad=0.0)
    
    # hyperparams = HyperParams(c0=1000000000, min_number_assignments=7, lambda_off=1.0, lambda_cen=0.0023008084069785772, lambda_rad=0.0)
    
    
    # hyperparams = HyperParams(c0=50000, min_number_assignments=7, lambda_off=0.81, lambda_cen=0.48, lambda_rad=0.34)

    
    atlas, pos_target_array, rad_target_array = build_atlas_and_data(atlas_data_dir, target_data_dir)

    # with open(f"{atlas_pickle_dir}/atlas.pkl", "wb") as f:
    #     pickle.dump(atlas, f)

    # with open("/home/daniel/pythonny/MPI/SeePelican/tests/full_worms/atlas_pickle/atlas2.pkl", "rb") as f:
    #     atlas = pickle.load(f)

    # with open("/home/daniel/pythonny/MPI/SeePelican/tests/full_worms/atlas_pickle/pos_target_array.pkl", "rb") as f:
    #     pos_target_array = pickle.load(f)

    accuracy, std, mininum_acc, maximum_acc = process_worms(atlas, pos_target_array[:], rad_target_array[:], dd_file_dir, solution_file_dir, matches_file_dir, key_file_dir, hyperparams, subgraph_size=subgraph_size, num_dd_file_workers=8, num_graph_match_workers=8)
    
    print(f"{subgraph_size} {accuracy:.6f} {std:.6f} {mininum_acc:.6f} {maximum_acc:.6f} {time.time() - start_time:.2f}")
    print(f"Subgraph size: {subgraph_size}")
    print(f"Mean accuracy: {accuracy:.6f}")
    print(f"Total time taken: {time.time() - start_time:.2f} seconds")
