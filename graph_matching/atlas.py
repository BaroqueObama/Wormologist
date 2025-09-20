import numpy as np

def compute_per_nucleus_cov(X):
    X_mean = np.mean(X, axis=0, keepdims=True)
    X_centered = X - X_mean # shape (n_worms, n_nuclei, 3)
    cov = np.einsum('wki,wkj->kij', X_centered, X_centered) / X.shape[0]
    return cov

def compute_offset_stats(position_array):
    n_worms, n_nuclei, _ = position_array.shape

    # Compute all pairwise offsets
    pos_i = position_array[:, :, None, :]
    pos_j = position_array[:, None, :, :]
    offsets = pos_i - pos_j # (n_worms, n_i, n_j, 3)

    mean_offsets = np.mean(offsets, axis=0)

    centered = offsets - mean_offsets[None, :, :, :]
    centered_flat = centered.reshape(n_worms, -1, 3)
    covs = np.einsum('wki,wkj->kij', centered_flat, centered_flat) / n_worms
    cov_offsets = covs.reshape(n_nuclei, n_nuclei, 3, 3)

    return mean_offsets, cov_offsets

def build_atlas(position_array, radius_array):
    mean_centers = np.mean(position_array, axis=0)
    cov_centers  = compute_per_nucleus_cov(position_array)

    mean_radii = np.mean(radius_array, axis=0)
    cov_radii  = compute_per_nucleus_cov(radius_array)

    mean_offsets, cov_offsets = compute_offset_stats(position_array)

    return {
        'mean_centers': mean_centers,
        'cov_centers': cov_centers,
        'mean_radii': mean_radii,
        'cov_radii': cov_radii,
        'mean_offsets': mean_offsets,
        'cov_offsets': cov_offsets
    }