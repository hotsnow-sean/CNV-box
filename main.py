import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("dir")
args = parser.parse_args()

import re

import numpy as np
from matplotlib import pyplot as plt
from pyod.models import mcd
from sklearn.mixture import BayesianGaussianMixture

from cnvutils import calc_result, combiningCNV, read_gt
from RDG import RDG

names = ["NA12878", "NA12891", "NA12892", "NA19238", "NA19239", "NA19240"]
files = {k: [] for k in names}

cnt = 0
for file in filter(re.compile(r"NA.*\.(bam|gt)").match, os.listdir(args.dir)):
    ext = os.path.splitext(file)[1]
    name = file.split(".", 2)[0]
    ll = files.get(name)
    if ll == None or len(ll) >= 2:
        continue
    cnt += 1
    if ext == "bam":
        ll.insert(0, os.path.join(args.dir, file))
    else:
        ll.append(os.path.join(args.dir, file))
    if cnt >= 2 * len(names):
        break
files = {k: v for k, v in files.items() if len(v) == 2}


def vbgmm(scores):
    """
    The variational Bayesian Gaussian mixture model (VBGMM)
    :param scores: ndarray (n_samples, 1), the outlier scores for all genome segments
    :return: binary labels, where 1 is outlier, 0 stands for inlier
    """
    clf = BayesianGaussianMixture(n_components=2)
    labels = clf.fit_predict(scores)
    outlier_label = np.argmax(clf.means_)
    labels = labels == outlier_label
    return labels.astype(int)


rdg = RDG()
rdg.read_fa({"21": "/mnt/e/LinuxSpace/CNV_DATA/reference/chr21.fa"})

candidate_support = np.arange(0.5, 0.91, 0.05)
results = {k: [] for k in files.keys()}

for k, v in files.items():
    rdg.binning(v[0])
    gt = read_gt(v[1], False)

    (rd, start, end), mode = rdg.segment(chr="21", method="no")
    for num in candidate_support:
        clf = mcd.MCD(support_fraction=num)
        clf.fit(rd.reshape(-1, 1))
        labels = vbgmm(clf.decision_scores_.reshape(-1, 1))

        result = combiningCNV(rd, start, end, labels, clf.location_[0])
        precision, sensitivity = calc_result(result, gt)
        results[k].append(2 * precision * sensitivity / (precision + sensitivity))


for k, v in results.items():
    plt.plot(candidate_support, v, label=k)

plt.legend()
plt.xlabel("Support Fraction")
plt.ylabel("F1-score")
plt.show()

# plt.xlim(0.0, 1.0)
# plt.ylim(0.0, 1.0)

# C = plt.contour(X, Y, Z, np.arange(0.1, 1.0, 0.1), colors="k", linestyles="--")
# plt.clabel(C)

# for i, (y, x) in enumerate(ans):
#     plt.scatter(x, y, c=colors[i], marker=markers[i], label=method[i])
# plt.legend()
# plt.xlabel("Sensitivity")
# plt.ylabel("Precision")
# plt.title(os.path.basename(v[0]).split(".", 2)[0])

# plt.show()
