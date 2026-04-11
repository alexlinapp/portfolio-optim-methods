from __future__ import annotations


def time_series_split_indices(n: int, train_frac: float, val_frac: float) -> tuple[slice, slice, slice]:
    """Non-overlapping chronological slices: train | val | test."""
    if not (0 < train_frac < 1 and 0 < val_frac < 1):
        raise ValueError("fractions must be in (0,1)")
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be < 1")
    t_end = int(n * train_frac)
    v_end = int(n * (train_frac + val_frac))
    return slice(0, t_end), slice(t_end, v_end), slice(v_end, n)
