from typing import Tuple
from matplotlib.axes import Axes
import numpy as np
import pandas as pd


__default_gain = "gain"
__default_loss = "loss"


def read_gt(gt_path: str, is_simulation: bool):
    """
    Read groundtruth file to a special DataFrame:
    Column's name: start, end, type
    Value's dtype: int, int, str(<gain>|<loss>)
    """
    if is_simulation:
        header = ["start", "end", "state"]
        gain_tag = "gain"
    else:
        header = ["start", "stop", "variant type"]
        gain_tag = "duplication"

    default_header = ["start", "end", "type"]

    gt = pd.read_csv(
        gt_path,
        sep="\t",
        usecols=header,
        dtype={header[0]: int, header[1]: int, header[2]: str},
    )
    gt.columns = default_header
    gt[default_header[2]] = gt[default_header[2]].map(
        lambda x: __default_gain if x == gain_tag else __default_loss
    )

    return gt


def calc_result(
    test: pd.DataFrame, gt, is_simulation: bool = False
) -> Tuple[float, float]:
    """
    Calculate the results through the test value and ground truth.

    Data format:
        Column's name: start, end, type
        Value's dtype: int, int, str(<gain>|<loss>)
    
    Parameters
    ----------
    test : DataFrame

    gt : DataFrame or str
        if its type is str, it will be as a groundtruth file path

    is_simulation : bool, default False
        used only gt is a str

    Returns
    -------
    precision: float
    sensitivity: float
    """
    if isinstance(gt, str):
        gt = read_gt(gt, is_simulation)

    if not isinstance(test, pd.DataFrame) or not isinstance(gt, pd.DataFrame):
        raise

    test.sort_values(by="start", inplace=True)
    gt.sort_values(by="start", inplace=True)

    match_cnt = i = j = 0
    test_cnt = test.agg(lambda x: x["end"] - x["start"] + 1, axis=1).agg(sum)
    gt_cnt = gt.agg(lambda x: x["end"] - x["start"] + 1, axis=1).agg(sum)

    while i < len(test) and j < len(gt):
        if test.iloc[i]["type"] == gt.iloc[j]["type"]:
            lo = max(test.iloc[i]["start"], gt.iloc[j]["start"])
            hi = min(test.iloc[i]["end"], gt.iloc[j]["end"])
            if lo <= hi:
                match_cnt += hi - lo + 1

        if test.iloc[i]["end"] < gt.iloc[j]["end"]:
            i += 1
        else:
            j += 1

    precision = match_cnt / test_cnt
    sensitivity = match_cnt / gt_cnt

    return precision, sensitivity


def draw_profile(
    ax: Axes,
    RD: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    gt: pd.DataFrame = None,
):
    RD = np.repeat(RD, 2)
    pos = np.c_[start, end].ravel()

    ax.plot(pos, RD)

    if gt is not None:
        for row in gt.itertuples():
            ax.axvspan(
                row.start,
                row.end,
                color="g" if row.type == __default_gain else "orange",
                alpha=0.5,
            )


def combiningCNV(
    rd: np.ndarray, start: np.ndarray, end: np.ndarray, labels: np.ndarray, mode: float
):
    """Calculate CNV by labels and mode.
    """
    index = labels == 1
    CNVRD = rd[index]
    CNVstart = start[index]
    CNVend = end[index]

    type = np.frompyfunc(lambda x: 2 if x > mode else 1, 1, 1)(CNVRD)

    for i in range(len(CNVRD) - 1):
        if CNVend[i] + 1 == CNVstart[i + 1] and type[i] == type[i + 1]:
            CNVstart[i + 1] = CNVstart[i]
            type[i] = 0

    index = type != 0
    CNVRD = CNVRD[index]
    CNVstart = CNVstart[index]
    CNVend = CNVend[index]
    CNVtype = type[index]
    CNVtype = [__default_gain if i == 2 else __default_loss for i in CNVtype]
    result = pd.DataFrame(
        {"start": CNVstart, "end": CNVend, "type": CNVtype, "RD": CNVRD}
    )
    return result
