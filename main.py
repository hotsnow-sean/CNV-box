import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("bam_path")
parser.add_argument("fa_path")
parser.add_argument("--gt")
parser.add_argument("-s", "--sim", action="store_true")
args = parser.parse_args()

from RDG import RDG
from cnvutils import read_gt, draw_profile
import matplotlib.pyplot as plt

chr = "chr21" if args.sim else "21"

rdg = RDG(chr=[chr])

(rd, start, end), mode = rdg.preprocess(args.bam_path, args.fa_path, chr=chr)

gt = None if args.gt is None else read_gt(args.gt, args.sim)

plt.figure(figsize=(15, 3))
draw_profile(plt.gca(), rd, start, end, gt)
plt.title(f"{os.path.basename(args.bam_path)} with cbs")
plt.tight_layout()
plt.savefig("test.png")
