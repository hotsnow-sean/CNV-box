import numpy as np
import pywt


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


def wave_seg(RD: np.ndarray, pos: np.ndarray, bin_size: int):
    import math

    def denoise(data, wavelet, noiseSigma):
        levels = 8
        WC = pywt.wavedec(data, wavelet, level=levels, mode="constant")
        noiseSigma = (2 / 0.6745) * np.mean(
            np.absolute(WC[-levels] - np.mean(WC[-levels]))
        )
        threshold = noiseSigma * math.sqrt(2 * math.log2(data.shape[0]))
        NWC = list(map(lambda x: pywt.threshold(x, threshold), WC))
        return pywt.waverec(NWC, wavelet, mode="constant")[: len(data)]

    newrd = denoise(RD, "haar", 0.1)
    newrd = newrd[: len(pos)]
    seg_index = []
    start = 0
    for i in range(1, len(newrd)):
        if not math.isclose(newrd[i], newrd[i - 1]):
            seg_index.append((start, i))
            start = i
    seg_index.append((start, len(newrd) - 1))
    seg_rd = np.empty(len(seg_index))
    seg_start = np.full(len(seg_index), 0)
    seg_end = np.full(len(seg_index), 0)
    for i in range(len(seg_index)):
        segment = RD[seg_index[i][0] : seg_index[i][1]]
        seg_rd[i] = np.mean(segment)
        seg_start[i] = pos[seg_index[i][0]] * bin_size + 1
        seg_end[i] = pos[seg_index[i][1] - 1] * bin_size + bin_size

    return seg_rd, seg_start, seg_end


def wave_cbs_seg(RD: np.ndarray, pos: np.ndarray, bin_size: int):
    WC = pywt.wavedec(RD, 'haar', level=8, mode="constant")
    WC[-2] = np.zeros_like(WC[-2])
    WC[-1] = np.zeros_like(WC[-1])
    nrd = pywt.waverec(WC, 'haar', mode='constant')
    return cbs_seg(nrd[:len(pos)], pos, bin_size)


def segment(RD: np.ndarray, pos: np.ndarray, bin_size: int, method: str = "cbs"):
    if method == "cbs":
        return cbs_seg(RD, pos, bin_size)
    elif method == "wave":
        return wave_seg(RD, pos, bin_size)
    elif method == "wave_cbs":
        return wave_cbs_seg(RD, pos, bin_size)
    else:
        return no_seg(RD, pos, bin_size)

