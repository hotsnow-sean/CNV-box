import numpy as np
import pywt
import math
from skimage.restoration import denoise_wavelet


def denoise(data, wavelet):
    levels = 5
    WC = pywt.wavedec(data, wavelet, level=levels, mode="constant")
    noiseSigma = (1 / 0.6745) * np.median(np.abs(WC[-1]))
    threshold = noiseSigma * math.sqrt(2 * math.log(data.shape[0]))
    NWC = list(map(lambda x: pywt.threshold(x, threshold, mode="garrote"), WC))
    return pywt.waverec(NWC, wavelet, mode="constant")[: len(data)]


def no_seg(RD: np.ndarray, pos: np.ndarray, bin_size: int):
    seg_rd = RD.copy()
    seg_start: np.ndarray = pos * bin_size + 1
    seg_end: np.ndarray = seg_start + bin_size - 1

    return seg_rd, seg_start, seg_end


def cbs_seg(RD: np.ndarray, pos: np.ndarray, bin_size: int):
    from cbseg import segment

    seg_index = segment(RD, shuffles=1000, p=0.05)

    seg_rd = np.empty(len(seg_index))
    seg_start = np.empty(len(seg_index), dtype=int)
    seg_end = np.empty(len(seg_index), dtype=int)

    for i, id in enumerate(seg_index):
        start = id.start
        end = id.end
        seg = RD[start:end]
        seg_rd[i] = np.mean(seg)
        seg_start[i] = pos[start] * bin_size + 1
        seg_end[i] = pos[end - 1] * bin_size + bin_size

    return seg_rd, seg_start, seg_end


def wave_seg(RD: np.ndarray, pos: np.ndarray, bin_size: int):
    # newrd = denoise(RD, "haar")
    newrd = denoise_wavelet(RD, method="VisuShrink")
    newrd = newrd[: len(pos)]
    seg_index = []
    start = 0
    for i in range(1, len(newrd)):
        if not math.isclose(newrd[i], newrd[i - 1]):
            seg_index.append((start, i))
            start = i
    seg_index.append((start, len(newrd)))
    seg_rd = np.empty(len(seg_index))
    seg_start = np.full(len(seg_index), 0)
    seg_end = np.full(len(seg_index), 0)
    for i, (start, end) in enumerate(seg_index):
        segment = RD[start:end]
        seg_rd[i] = np.mean(segment)
        seg_start[i] = pos[start] * bin_size + 1
        seg_end[i] = pos[end - 1] * bin_size + bin_size

    return seg_rd, seg_start, seg_end


def wave_cbs_seg(RD: np.ndarray, pos: np.ndarray, bin_size: int):
    nrd = denoise_wavelet(RD, method="VisuShrink", wavelet_levels=2)
    return cbs_seg(nrd[: len(pos)], pos, bin_size)


def segment(RD: np.ndarray, pos: np.ndarray, bin_size: int, method: str = "cbs"):
    if method == "cbs":
        return cbs_seg(RD, pos, bin_size)
    elif method == "wave":
        return wave_seg(RD, pos, bin_size)
    elif method == "wave_cbs":
        return wave_cbs_seg(RD, pos, bin_size)
    else:
        return no_seg(RD, pos, bin_size)

