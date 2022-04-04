import numpy as np


def no_seg(RD: np.ndarray, pos: np.ndarray, bin_size: int):
    seg_rd = RD.copy()
    seg_start: np.ndarray = pos * bin_size + 1
    seg_end: np.ndarray = seg_start + bin_size - 1

    return seg_rd, seg_start, seg_end


def cbs_seg(RD: np.ndarray, pos: np.ndarray, bin_size: int):
    from cbs import segment

    seg_index = segment(RD)

    seg_rd = np.empty(len(seg_index))
    seg_start = np.empty(len(seg_index), dtype=int)
    seg_end = np.empty(len(seg_index), dtype=int)

    for i, (start, end) in enumerate(seg_index):
        seg = RD[start:end]
        seg_rd[i] = np.mean(seg)
        seg_start[i] = pos[start] * bin_size + 1
        seg_end[i] = pos[end - 1] * bin_size + bin_size

    return seg_rd, seg_start, seg_end


def segment(RD: np.ndarray, pos: np.ndarray, bin_size: int, method: str = "cbs"):
    if method == "no":
        return no_seg(RD, pos, bin_size)
    elif method == "cbs":
        return cbs_seg(RD, pos, bin_size)
    else:
        return None, None, None

