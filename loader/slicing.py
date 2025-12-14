from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.decomposition import PCA

def get_slice_indices(
    points: np.ndarray,
    n_slices: int = 40,
    slice_thickness: float = 0.005,
    shift: float = 0.0,
    *,
    crop_axis: Optional[str] = None,   # 'x', 'y', 'random', or None
    crop_side: str = "positive",
    crop_fraction: float = 0.0,
    random_state: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, List[int], np.ndarray]:
    """
    Slice a worm‑shaped point cloud into thin slabs and optionally remove one half
    of each slice. The main (AP) axis is estimated via PCA; crop_axis controls
    whether you drop the positive or negative side of x (LR), y (DV), a random
    in‑plane direction, or nothing at all. crop_fraction retains a thin band
    near the centreline on the removed side.

    Parameters
    ----------
    points : (N, 3) array
        Cartesian nucleus centres.  Axes are assumed to be (x=LR, y=DV, z=AP).
    n_slices : int
        Number of slices along the worm’s length.
    slice_thickness : float
        Thickness of each axial slab (distance along the AP axis).
    shift : float
        Uniform shift applied to all slice centres along the AP axis.
    crop_axis : {"x", "y", "random", None}
        Direction along which to crop within each slice:
          • None: no cropping.
          • 'x' : remove LR half (positive or negative).
          • 'y' : remove DV half.
          • 'random': remove half along a random direction in the xy‑plane.
    crop_side : {"positive", "negative"}
        Which half to remove (“positive” removes values > 0 and keeps ≤ 0).
    crop_fraction : float
        Fraction of the removed side’s span to retain near zero. 0.0 keeps
        nothing; 1.0 keeps the entire removed side.
    random_state : np.random.Generator, optional
        Source of randomness for 'random' crop_axis; use to make behaviour reproducible.

    Returns
    -------
    indices : (M,) array of int
        Sorted indices of nuclei that survived slicing and cropping.
    per_slice_counts : list of int
        Number of nuclei kept in each of the n_slices axial slabs.
    projected_coords : (M, 3) array
        The 3D coordinates projected onto their slice planes.
    """
    # Step 1: fit PCA to find the AP axis; use only PC0 for slicing.
    pca = PCA(n_components=3)
    pca.fit(points)
    axial_axis = pca.components_[0]           # unit vector for AP
    mean = pca.mean_
    t = (points - mean) @ axial_axis          # scalar position along AP for each point

    # Centre the cloud once; used for all in‑plane coordinate calculations.
    centered = points - mean
    slice_coord: Optional[np.ndarray] = None  # will hold signed distances for cropping

    # Step 2: choose in‑plane direction based on crop_axis.
    if crop_axis is None:
        # no cropping requested
        pass
    elif crop_axis == "x":
        # crop along LR; positive x values lie on one side, negative on the other
        slice_coord = centered[:, 0]
    elif crop_axis == "y":
        # crop along DV
        slice_coord = centered[:, 1]
    elif crop_axis == "random":
        # sample a random unit vector in the xy‑plane (perpendicular to z)
        rng = random_state or np.random.default_rng()
        angle = rng.uniform(0.0, 2.0 * np.pi)
        cos_orient = np.cos(angle)
        sin_orient = np.sin(angle)
        global_x_unit = np.array([1.0, 0.0, 0.0], dtype=points.dtype)
        global_y_unit = np.array([0.0, 1.0, 0.0], dtype=points.dtype)
        rand_vec = cos_orient * global_x_unit + sin_orient * global_y_unit
        slice_coord = centered @ rand_vec
    else:
        raise ValueError(f"crop_axis must be 'x', 'y', 'random' or None; got {crop_axis}")

    # Step 3: create slice centres along AP and assign points to slabs.
    centres = np.linspace(t.min(), t.max(), n_slices) + shift
    selected: Dict[int, Dict[str, Any]] = {}
    half_thickness = slice_thickness / 2.0

    for slice_idx, centre_val in enumerate(centres):
        mask = np.abs(t - centre_val) <= half_thickness
        hits = np.where(mask)[0]

        # Apply optional in‑plane cropping.
        if slice_coord is not None and hits.size > 0:
            vals = slice_coord[hits]
            if crop_side == "positive":
                keep_mask = vals <= 0.0
                if crop_fraction > 0.0:
                    pos_vals = vals[vals > 0.0]
                    if pos_vals.size > 0:
                        span = pos_vals.max()
                        allowed = span * crop_fraction
                        keep_mask |= (vals > 0.0) & (vals <= allowed)
            else:
                keep_mask = vals >= 0.0
                if crop_fraction > 0.0:
                    neg_vals = vals[vals < 0.0]
                    if neg_vals.size > 0:
                        span = abs(neg_vals.min())
                        allowed = span * crop_fraction
                        keep_mask |= (vals < 0.0) & (vals >= -allowed)
            hits = hits[keep_mask]

        # Project survivors onto their slice plane; deduplicate by keeping the nearest slice.
        for idx in hits:
            offset = t[idx] - centre_val
            abs_offset = abs(offset)
            prev = selected.get(idx)
            if prev is None or abs_offset < prev["abs_offset"]:
                projected = points[idx] - offset * axial_axis
                selected[idx] = {
                    "abs_offset": abs_offset,
                    "projected": projected,
                    "slice_index": slice_idx,
                }

    if not selected:
        return np.array([], dtype=int), [0] * n_slices, np.zeros((0, 3), dtype=points.dtype)

    per_slice_counts: List[int] = [0] * n_slices
    for v in selected.values():
        per_slice_counts[v["slice_index"]] += 1

    indices = np.array(sorted(selected.keys()), dtype=int)
    projected_coords = np.stack([selected[i]["projected"] for i in indices], axis=0)

    return indices, per_slice_counts, projected_coords