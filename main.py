import argparse
import os

import numpy as np

# parse command line argument
parser = argparse.ArgumentParser()
parser.add_argument("bam_path")
parser.add_argument("fa_path")
parser.add_argument("--gt")
parser.add_argument("-s", "--sim", action="store_true")
args = parser.parse_args()

from RDG import RDG
from cnvutils import read_gt, draw_profile
import matplotlib.pyplot as plt

# simulitation or not
chr = "chr21" if args.sim else "21"

# binning
rdg = RDG(chr=[chr])
rdg.binning(args.bam_path, {chr: args.fa_path})

np.save("rddata", rdg.bin_profile_[chr].rd)
np.save("posdata", rdg.bin_profile_[chr].pos)

# # read groundtruth
# gt = None if args.gt is None else read_gt(args.gt, args.sim)

# # draw
# fig = plt.figure(figsize=(15, 6))

# method = ["cbs", "wave"]

# for i, m in enumerate(method):
#     (rd, start, end), mode = rdg.segment(chr=chr, method=m)

#     ax = fig.add_subplot(2, 1, i + 1)
#     draw_profile(ax, rd, start, end, gt)
#     ax.set_title(f"{os.path.basename(args.bam_path)} with {m} segment")

# plt.tight_layout()
# plt.show()
