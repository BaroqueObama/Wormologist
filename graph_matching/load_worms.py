import pandas as pd
import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.spatial import KDTree
from scipy.optimize import minimize_scalar


def load_worm_data(file_paths, usecols=[i for i in range(1, 8)], dtype=np.float32, NUM_NUCLEI=558, header=None, delimiter=" ", sort=True):
    position_array = np.zeros((len(file_paths), NUM_NUCLEI, 3)) # length of file_paths should be number of worms
    radius_array = np.zeros((len(file_paths), NUM_NUCLEI, 3))

    for i, file_path in enumerate(file_paths):
        df = pd.read_csv(file_path, usecols=usecols, header=header, delimiter=delimiter)
        df[usecols[1:]] = df[usecols[1:]].astype(dtype)
        df[usecols[0]] = df[usecols[0]].str.capitalize()
        if sort:
            df = df.sort_values(by=usecols[0])
        
        worm_point = df[usecols[1:4]].to_numpy()
        position_array[i] = worm_point
        worm_radii = df[usecols[4:]].to_numpy()
        radius_array[i] = worm_radii

    return position_array, radius_array


def center_and_scale(X):
    X_centered = X - X.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(X_centered)
    return X_centered / scale if scale > 0 else X_centered


def procrustes_transform(ref, target):
    ref_cs = center_and_scale(ref)
    target_cs = center_and_scale(target)

    U, _, Vt = np.linalg.svd(target_cs.T @ ref_cs)
    R = U @ Vt
    aligned = target_cs @ R
    return aligned


def align_all_worms_with_transforms(X, ref):
    aligned = []
    for worm in X:
        aligned_worm = procrustes_transform(ref, worm)
        aligned.append(aligned_worm)
    return np.array(aligned)


def generalized_procrustes(X, max_iter=10, tol=1e-6):
    mean_shape = center_and_scale(np.mean(X, axis=0))

    for _ in range(max_iter):
        aligned = align_all_worms_with_transforms(X, mean_shape)
        new_mean = center_and_scale(np.mean(aligned, axis=0))
        if np.linalg.norm(new_mean - mean_shape) < tol:
            break
        mean_shape = new_mean

    return aligned


# Worm Alignment Scripts

def center(X):
    return X - X.mean(axis=0), X.mean(axis=0)


def pca_axes(Xc):
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt.T
    if np.linalg.norm(V) > 0 and np.linalg.det(V) < 0:
        V[:, 2] *= -1
    return V


def rot_matrix_from_a_to_b(a, b, eps=1e-9):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < eps or norm_b < eps:
        return np.eye(3)
    
    a = a / norm_a
    b = b / norm_b
    
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    
    if s < eps:
        if c > 0:
            return np.eye(3)  # Same direction
        else:
            # Find perpendicular axis
            if abs(a[0]) < 0.9:
                perp = np.array([1.0, 0.0, 0.0])
            else:
                perp = np.array([0.0, 1.0, 0.0])
            v = np.cross(a, perp)
            v = v / np.linalg.norm(v)
            # 180 degree rotation around perpendicular axis
            return 2 * np.outer(v, v) - np.eye(3)
    
    # Rodrigues' rotation formula
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    R = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s * s))
    
    return R


def rot_about_axis(k, theta, eps=1e-9):
    # Properly normalize the axis vector
    norm_k = np.linalg.norm(k)
    if norm_k < eps:
        return np.eye(3)
    
    k = k / norm_k
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def find_symmetry_axis(points):
    if len(points) < 10:
        return 0.0
    tree = KDTree(points)
    def score(a):
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        reflected = (points @ R.T) * np.array([1, -1])
        return np.mean(tree.query(reflected @ R)[0])
    try:
        angle = minimize_scalar(score, bounds=(0, np.pi), method='bounded').x
        return angle
    except:
        return 0.0


def get_head_region_points(X, v1, head_fraction=1.0):
    proj_v1 = X @ v1
    
    pos_count = np.sum(proj_v1 > 0)
    neg_count = np.sum(proj_v1 < 0)
    head_side = 1 if pos_count > neg_count else -1
    
    if head_side > 0:
        head_mask = (proj_v1 > 0)
    else:
        head_mask = (proj_v1 < 0)
    
    head_points_all = X[head_mask]
    head_proj_all = head_points_all @ v1
    sorted_indices = np.argsort(np.abs(head_proj_all))[::-1]
    
    n_total = len(X)
    n_head = max(10, int(n_total * head_fraction))
    selected_head_points = head_points_all[sorted_indices[:n_head]]
    
    return selected_head_points, head_side


def get_dv_axis(X, v1, v2, v3, head_fraction=1.0, use_head_region=True):
    if use_head_region and head_fraction < 1.0:
        analysis_points, _ = get_head_region_points(X, v1, head_fraction=head_fraction)
    else:
        analysis_points = X
    
    if len(analysis_points) < 10:
        analysis_points = X
    
    proj_v2 = analysis_points @ v2
    proj_v3 = analysis_points @ v3
    proj_2d = np.column_stack((proj_v2, proj_v3))
    
    angle = find_symmetry_axis(proj_2d)
    lr = np.cos(angle) * v2 + np.sin(angle) * v3
    dv = -np.sin(angle) * v2 + np.cos(angle) * v3
    
    return dv, lr


def compare_density(X, axis, head_fraction=1.0, use_head_region=True):
    if use_head_region and head_fraction < 1.0:
        analysis_points, head_side = get_head_region_points(X, axis, head_fraction=head_fraction)
    else:
        analysis_points = X
        head_side = 1
    
    if len(analysis_points) == 0:
        return 0
    
    proj = analysis_points @ axis
    if len(proj) < 5:
        return 0
        
    hist, bins = np.histogram(proj, bins=min(10, len(proj)))
    if len(hist) < 2:
        return 0
        
    mid = len(hist) // 2
    left = np.sum(hist[:mid])
    right = np.sum(hist[mid:])
    
    return 1 if left > right else -1


def align_pointcloud_with_atlas(atlas, target, scale_to_unit_fro=True, eps=1e-9, head_fraction=0.95):
    A0, cA = center(atlas)
    T0, cT = center(target)

    VA = pca_axes(A0)
    VT = pca_axes(T0)
    v1A, v2A, v3A = VA.T
    v1T, v2T, v3T = VT.T

    use_head_region = (head_fraction < 1.0)
    dvA, lrA = get_dv_axis(A0, v1A, v2A, v3A, head_fraction=head_fraction, use_head_region=use_head_region)
    atlas_density_sign = compare_density(A0, dvA, head_fraction=head_fraction, use_head_region=use_head_region)

    R1 = rot_matrix_from_a_to_b(v1T, v1A, eps=eps)
    T1 = (T0 @ R1.T)
    v2T1, v3T1 = R1 @ v2T, R1 @ v3T

    dvT, lrT = get_dv_axis(T1, v1A, v2T1, v3T1, head_fraction=head_fraction, use_head_region=use_head_region)
    target_density_sign = compare_density(T1, dvT, head_fraction=head_fraction, use_head_region=use_head_region)

    theta = np.arctan2(np.linalg.norm(np.cross(dvT, dvA)), np.dot(dvT, dvA))
    R2 = rot_matrix_from_a_to_b(dvT, dvA, eps=eps)
    T2 = T1 @ R2.T

    new_dv = R2 @ dvT
    new_density_sign = compare_density(T2, new_dv, head_fraction=head_fraction, use_head_region=use_head_region)
    
    if new_density_sign != atlas_density_sign:
        R_flip = rot_about_axis(v1A, np.pi, eps=eps)
        T2 = T2 @ R_flip.T
        R2 = R_flip @ R2

    s = 1.0
    if scale_to_unit_fro:
        s = 1.0 / (np.linalg.norm(T2, 'fro') + eps)
        T2 = T2 * s

    aligned = T2 + cA
    R = R2 @ R1
    t = cA - (R @ (s * cT))

    return aligned


def align_worms_to_standard_axes(worms, scale_to_unit_fro=True, head_fraction=0.95, use_head_region=False):
    # Align worms to axes (Z=AP, X=LR, Y=DV)
    avg_worm = np.mean(worms, axis=0)
    avg_worm_centered, avg_center = center(avg_worm)
    
    V_avg = pca_axes(avg_worm_centered)
    v1_avg, v2_avg, v3_avg = V_avg.T
    
    dv_avg, lr_avg = get_dv_axis(avg_worm_centered, v1_avg, v2_avg, v3_avg, head_fraction=head_fraction, use_head_region=use_head_region)
    
    target_v1 = np.array([0, 0, 1])
    target_lr = np.array([1, 0, 0])
    target_dv = np.array([0, 1, 0])
    
    R1 = rot_matrix_from_a_to_b(v1_avg, target_v1)
    
    # Check AP axis density orientation after initial alignment
    avg_worm_after_r1 = avg_worm_centered @ R1.T
    ap_density_sign = compare_density(avg_worm_after_r1, target_v1, use_head_region=False)
    
    # If denser side faces positive direction, flip AP axis
    if ap_density_sign == -1:
        R_ap_flip = rot_about_axis(target_lr, np.pi)
        R1 = R_ap_flip @ R1
        avg_worm_after_r1 = avg_worm_centered @ R1.T
    
    dv_rot1 = R1 @ dv_avg
    lr_rot1 = R1 @ lr_avg
    
    theta = np.arctan2(np.linalg.norm(np.cross(dv_rot1, target_dv)), np.dot(dv_rot1, target_dv))
    R2 = rot_matrix_from_a_to_b(dv_rot1, target_dv)
    
    R_avg = R2 @ R1
    
    avg_worm_aligned = (avg_worm_centered @ R_avg.T)
    
    # compare_density returns 1 if negative side is denser, -1 if positive side is denser
    dv_density_sign = compare_density(avg_worm_aligned, target_dv, use_head_region=False)
    
    # We want denser side on negative Y, so if positive side is denser (sign = -1), flip
    if dv_density_sign == -1:
        # Flip around Z axis (AP axis) to reverse the DV orientation
        R_flip = rot_about_axis(target_v1, np.pi)
        R_avg = R_flip @ R_avg
    
    aligned_worms = np.zeros_like(worms)
    transforms = []
    
    for i, worm in enumerate(worms):
        worm_centered, worm_center = center(worm)
        worm_aligned = worm_centered @ R_avg.T
        
        s = 1.0
        if scale_to_unit_fro:
            s = 1.0 / (np.linalg.norm(worm_aligned, 'fro') + 1e-9)
            worm_aligned = worm_aligned * s
        
        aligned_worms[i] = worm_aligned
        transforms.append((R_avg, avg_center - (R_avg @ (s * worm_center)), s))
    
    return aligned_worms, transforms


def align_worm_using_lr_points(worm, left_points, right_points, scale_to_unit_fro=True, eps=1e-9):
    worm = np.asarray(worm)
    left_points = np.asarray(left_points)
    right_points = np.asarray(right_points)
    
    worm_centered, worm_center = center(worm)
    left_centered = left_points - worm_center
    right_centered = right_points - worm_center
    
    V = pca_axes(worm_centered)
    v1, v2, v3 = V.T 
    
    left_center = np.mean(left_centered, axis=0)
    right_center = np.mean(right_centered, axis=0)
    
    lr_vector = left_center - right_center
    lr_norm = np.linalg.norm(lr_vector)
    if lr_norm < eps:
        raise ValueError("Left and right points are too close or coincident")
    lr_axis = lr_vector / lr_norm
    
    dot_product = np.abs(np.dot(v1, lr_axis))
    if dot_product > 0.99:  # Axes are nearly parallel
        # Try using second principal component as AP axis instead
        v1 = v2
        dot_product = np.abs(np.dot(v1, lr_axis))
        if dot_product > 0.99:
            raise ValueError("LR axis is parallel to principal axes - cannot determine unique orientation")
    
    # Make LR axis perpendicular to AP axis
    lr_perp = lr_axis - np.dot(lr_axis, v1) * v1
    lr_perp_norm = np.linalg.norm(lr_perp)
    if lr_perp_norm < eps:
        raise ValueError("LR axis is parallel to AP axis")
    lr_perp = lr_perp / lr_perp_norm
    
    dv_axis = np.cross(v1, lr_perp)
    dv_axis = dv_axis / np.linalg.norm(dv_axis)
    
    target_ap = np.array([0, 0, 1])  # AP -> Z axis
    target_lr = np.array([1, 0, 0])  # LR -> X axis (left=positive)
    target_dv = np.array([0, 1, 0])  # DV -> Y axis
    
    R1 = rot_matrix_from_a_to_b(v1, target_ap, eps=eps)
    
    lr_rot1 = R1 @ lr_perp
    dv_rot1 = R1 @ dv_axis
    
    # Now rotate the LR axis to X axis (while keeping AP/Z fixed)
    angle = np.arctan2(lr_rot1[1], lr_rot1[0])
    R2 = rot_about_axis(target_ap, -angle, eps=eps)
    
    R = R2 @ R1
    
    worm_aligned = worm_centered @ R.T
    
    # Check AP axis density orientation
    ap_density_sign = compare_density(worm_aligned, target_ap, use_head_region=False)
    
    # If denser side faces positive direction, flip AP axis
    if ap_density_sign == -1:
        R_ap_flip = rot_about_axis(target_lr, np.pi, eps=eps)
        R = R_ap_flip @ R
        worm_aligned = worm_aligned @ R_ap_flip.T
    
    s = 1.0
    if scale_to_unit_fro:
        s = 1.0 / (np.linalg.norm(worm_aligned, 'fro') + eps)
        worm_aligned = worm_aligned * s
    
    t = -(R @ (s * worm_center))
    
    return worm_aligned, (R, t, s)


def invert_perms(scramble):
    num_worms, num_nuclei = scramble.shape
    inv = np.empty_like(scramble)
    inv[np.arange(num_worms)[:, None], scramble] = np.arange(num_nuclei)
    return inv


def load_atlas_and_target_worms(pickle_dir, iteration_number):
    import pickle
    with open(f"{pickle_dir}/atlas.pkl", "rb") as f:
        atlas = pickle.load(f)

    with open(f"{pickle_dir}/pos_target_array_{iteration_number}.pkl", "rb") as f:
        pos_target_array = pickle.load(f)

    with open(f"{pickle_dir}/rad_target_array_{iteration_number}.pkl", "rb") as f:
        rad_target_array = pickle.load(f)
        
    with open(f"{pickle_dir}/scramble_{iteration_number}.pkl", "rb") as f:
        scramble = pickle.load(f)

    inv_scramble = invert_perms(scramble)
    pos_target_array = np.take_along_axis(pos_target_array, inv_scramble[:, :, None], axis=1)
    rad_target_array = np.take_along_axis(rad_target_array, inv_scramble[:, :, None], axis=1)

    return atlas, pos_target_array, rad_target_array
