import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment


def calculate_unary_costs(atlas, target_position, target_radii, hyperparams):
    n_atlas  = atlas['mean_centers'].shape[0]
    n_target = target_position.shape[0]

    delta_cen = target_position[None, :, :] - atlas['mean_centers'][:, None, :]
    delta_rad = target_radii[None, :, :]   - atlas['mean_radii'][:,  None, :]

    cov_cen_inv = np.linalg.pinv(atlas['cov_centers'])
    cov_rad_inv = np.linalg.pinv(atlas['cov_radii'])

    dcen = np.einsum('ism,imn,isn->is', delta_cen, cov_cen_inv, delta_cen)
    drad = np.einsum('ism,imn,isn->is', delta_rad, cov_rad_inv, delta_rad)

    unary_total = hyperparams.lambda_cen * dcen + hyperparams.lambda_rad * drad - hyperparams.c0

    if not hyperparams.only_hungarian and hyperparams.min_number_assignments > 0:
        k_row = min(hyperparams.min_number_assignments, n_target)  # per-atlas
        k_col = min(hyperparams.min_number_assignments, n_atlas)   # per-target

        mask = np.zeros((n_atlas, n_target), dtype=bool)

        part_row = np.argpartition(dcen, k_row-1, axis=1)[:, :k_row]
        rows = np.arange(n_atlas)[:, None]
        mask[rows, part_row] = True

        part_col = np.argpartition(dcen, k_col-1, axis=0)[:k_col, :]
        cols = np.arange(n_target)[None, :]
        mask[part_col, cols] = True
    elif not hyperparams.only_hungarian:
        mask = dcen < hyperparams.dcen_threshold

    if hyperparams.ensure_lap or hyperparams.only_hungarian:
        if hyperparams.only_hungarian:
            mask = np.zeros((n_atlas, n_target), dtype=bool)
        row_ind, col_ind = linear_sum_assignment(unary_total)
        mask[row_ind, col_ind] = True

    idx_i, idx_s = np.where(mask)
    costs = unary_total[idx_i, idx_s]
    assignment_ids = np.arange(len(idx_i), dtype=int)
    return assignment_ids, idx_i, idx_s, costs


def calculate_pairwise_costs(atlas, target_position, assignment_ids, idx_i, idx_s, hyperparams):
    k = len(assignment_ids)
    pos_s = target_position[idx_s]

    atlas_idx, target_idx = np.triu_indices(k, k=1)

    valid = (idx_i[atlas_idx] != idx_i[target_idx]) & (idx_s[atlas_idx] != idx_s[target_idx]) # Distinct source and target nodes
    atlas_idx = atlas_idx[valid]
    target_idx = target_idx[valid]

    delta = (pos_s[atlas_idx] - pos_s[target_idx])
    mean_offsets = atlas['mean_offsets'][idx_i[atlas_idx], idx_i[target_idx]]
    cov_inv = np.linalg.pinv(atlas['cov_offsets'][idx_i[atlas_idx], idx_i[target_idx]])

    diff = mean_offsets - delta
    doff = np.einsum('ijm,ijmn,ijn->i', diff[:, None, :], cov_inv[:, None, :, :], diff[:, None, :])
    
    if hyperparams.only_hungarian:
        pairwise_costs = 0 * doff
    else:
        pairwise_costs = hyperparams.lambda_off * doff

    src_ids = assignment_ids[atlas_idx]
    tgt_ids = assignment_ids[target_idx]

    return src_ids, tgt_ids, pairwise_costs


def write_dd_file(filename, n_atlas, n_target, assignment_ids, idx_i, idx_s, src_ids, tgt_ids, unary_costs, pairwise_costs):
    unary_dict = dict(zip(assignment_ids, unary_costs)) # Map assignment id to unary cost

    with open(filename, "w") as f:
        f.write(f"p {n_atlas} {n_target} {len(assignment_ids)} {len(src_ids)}\n")

        for a_id, i, s, c in zip(assignment_ids, idx_i, idx_s, unary_costs):
            assert np.isscalar(c) and np.isfinite(c), f"Bad unary cost: {c}"
            f.write(f"a {a_id} {i} {s} {c:.6f}\n")

        for x_is, x_jt, pair_c in zip(src_ids, tgt_ids, pairwise_costs):
            if x_is != x_jt and x_is in unary_dict and x_jt in unary_dict:
                if np.isfinite(pair_c):
                    f.write(f"e {x_is} {x_jt} {pair_c:.6f}\n")


def build_dd_file(atlas, target_position, target_radii, filename, hyperparams):
    n_target = target_position.shape[0]

    assignment_ids, idx_i, idx_s, unary_costs = calculate_unary_costs(atlas, target_position, target_radii, hyperparams)
    src_ids, tgt_ids, pairwise_costs = calculate_pairwise_costs(atlas, target_position, assignment_ids, idx_i, idx_s, hyperparams)
    
    unique_target_values, target_mapped = np.unique(idx_s, return_inverse=True)
    n_atlas = atlas['mean_centers'].shape[0]
    n_target = unique_target_values.size
    
    write_dd_file(
        filename,
        n_atlas=n_atlas, n_target=n_target,
        assignment_ids=assignment_ids, idx_i=idx_i, idx_s=target_mapped,
        src_ids=src_ids, tgt_ids=tgt_ids,
        unary_costs=unary_costs, pairwise_costs=pairwise_costs
    )
    
    return idx_s, target_mapped


def take_subgraph(pos_target_worm, rad_target_worm, subgraph_size):
    # indexes = np.random.choice(np.arange(pos_target_worm.shape[0]), size=subgraph_size, replace=False)
    indexes = get_slice_indices(pos_target_worm, total_nuclei=subgraph_size)
    pos_target_worm_sample = pos_target_worm[indexes]
    rad_target_worm_sample = rad_target_worm[indexes]
    return pos_target_worm_sample, rad_target_worm_sample, indexes


def create_solution_array(n_atlas, idx_s, target_mapped):
    solution_array = np.full(n_atlas, -1, dtype=int)
    solution_array[idx_s] = target_mapped
    return solution_array


def build_dd_and_solution(i, atlas, pos_target_worm, rad_target_worm, output_dir, solution_dir, key_dir, subgraph_size, hyperparams):
    output_path = f"{output_dir}/worm_{i:03d}.dd"
    solution_path = f"{solution_dir}/worm_{i:03d}.txt"
    key_path = f"{key_dir}/worm_{i:03d}.txt"

    pos_target_worm_sample, rad_target_worm_sample, indexes = take_subgraph(pos_target_worm, rad_target_worm, subgraph_size)
    np.savetxt(key_path, indexes, fmt="%d")

    idx_s, target_mapped = build_dd_file(atlas, pos_target_worm_sample, rad_target_worm_sample, output_path, hyperparams)
    solution_array = create_solution_array(atlas['mean_centers'].shape[0], indexes[idx_s], target_mapped)
    np.savetxt(solution_path, solution_array, fmt="%d")
    
    return output_path


def build_dd_file_all_worms(atlas, pos_target_array, rad_target_array, output_dir, solution_dir, key_dir, hyperparams, subgraph_size=558, num_workers=8):
    n_worms = pos_target_array.shape[0]
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(build_dd_and_solution, i, atlas, pos_target_array[i], rad_target_array[i], output_dir, solution_dir, key_dir, subgraph_size, hyperparams) for i in range(n_worms)]

        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                print(f"Error processing worm: {e}")


def get_slice_indices(points, n_slices=40, total_nuclei=110, tol=1e-2, max_iter=100):
    if total_nuclei > len(points):
        raise ValueError("Requested more nuclei than available.")

    pca = PCA(n_components=1)
    t = pca.fit_transform(points).ravel()
    centers = np.linspace(t.min(), t.max(), n_slices)

    def get_total_selected(thickness):
        selected = set()
        for c in centers:
            mask = np.abs(t - c) <= thickness / 2
            selected.update(np.where(mask)[0])
        return selected

    lo, hi = 0, t.max() - t.min()
    
    for i in range(max_iter):
        mid = (lo + hi) / 2
        count = len(get_total_selected(mid))
        if count < total_nuclei:
            lo = mid
        else:
            hi = mid
        if abs(count - total_nuclei) <= tol:
            break
    
    selected = np.array(sorted(get_total_selected(hi)))

    if len(selected) > total_nuclei:
        selected = selected[:total_nuclei]
    elif len(selected) < total_nuclei:
        raise RuntimeError("Failed to find exactly N nuclei.")
    return selected




