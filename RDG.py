from functools import reduce
import math
import os
from typing import Dict, Sequence
import warnings
import numpy as np
import pysam
from Bio import SeqIO


def mode_rd(RD: np.ndarray, bin_size: int):
    new_RD: np.ndarray = np.rint(RD * bin_size)
    new_RD = new_RD.astype(int)
    new_RD = new_RD[new_RD != 0]
    count = np.bincount(new_RD)

    if len(count) < 50:
        return np.mean(new_RD) / bin_size

    count = np.convolve(count, np.ones(50), "valid") / 50
    mode_min = np.argmax(count)
    return (mode_min + 25) / bin_size


def gc_correct(RD: np.ndarray, Gc: np.ndarray):
    bin_count = np.bincount(Gc)
    global_rd_ave = np.mean(RD)
    for i in range(len(RD)):
        if bin_count[Gc[i]] < 2:
            continue
        mean = np.mean(RD[Gc == Gc[i]])
        if not math.isclose(mean, 0.0):
            RD[i] *= global_rd_ave / mean
    return RD


class RDG:
    """RD profile generator.
    """

    class RDdata:
        def __init__(self, num: int) -> None:
            self.pos: np.ndarray = np.arange(num)
            self.rd: np.ndarray = np.zeros(num)
            self.mode: float = 0.0

    class RefData:
        def __init__(self, val: str) -> None:
            self.value: str = val
            self.valid: np.ndarray = None
            self.gc: np.ndarray = None
            self.__last_bin_size: int = 0

        def __len__(self):
            return len(self.value)

        def stat(self, bin_size: int):
            if self.__last_bin_size == bin_size:
                return
            self.__last_bin_size = bin_size

            bin_num = len(self.value) // bin_size
            self.valid = np.full(bin_num, True)
            self.gc = np.full(bin_num, 0)

            for i in range(bin_num):
                start = i * bin_size
                end = start + bin_size
                self.valid[i] = all(
                    self.value[bp] in "AGCTagct" for bp in range(start, end)
                )
                self.gc[i] = round(
                    reduce(
                        lambda c, bp: c + (1 if self.value[bp] in "GCgc" else 0),
                        range(start, end),
                        0,
                    )
                    * 1000
                    / bin_size
                )

    def __init__(self, chr: Sequence[str] = ["21"]) -> None:
        self.__chrs = [v for v in set(chr) if isinstance(v, str)]
        self.__bin_size: int = 0
        self.bin_profile_: Dict[str, RDG.RDdata] = {}
        self.ref_profile_: Dict[str, RDG.RefData] = {}

    def preprocess(
        self,
        bam_path: str,
        fa_path: str,
        bin_size: int = 1000,
        chr: str = "21",
        seg: str = "cbs",
    ):
        if self.__bin_size <= 0:
            self.binning(bam_path, {chr: fa_path}, bin_size)
        return self.segment(chr, seg)

    def segment(self, chr: str, method: str = "cbs"):
        if self.__bin_size <= 0:
            warnings.warn("Please call binning method before segment.")
            return (None, None, None), 0.0
        if chr not in self.bin_profile_:
            warnings.warn("Please check chr id.")
            return (None, None, None), 0.0

        from segutils import segment

        rd = self.bin_profile_[chr]

        print(f"> segment start, chrID: {chr}, method: {method}")
        return segment(rd.rd, rd.pos, self.__bin_size, method), rd.mode

    def binning(self, bam_path: str, fa_path: dict = None, bin_size: int = 1000):
        # clear old value
        self.bin_profile_.clear()
        self.__bin_size = 0

        # read references
        if fa_path is not None:
            self.read_fa(fa_path)

        # check references avaliabled
        check_result = self.__check_bam_ref(bam_path)
        if check_result is None:
            return

        # init bin profile
        samfile, refs = check_result
        bin_nums = {k: len(v) // bin_size for k, v in refs.items()}
        self.bin_profile_ = {k: RDG.RDdata(v) for k, v in bin_nums.items()}

        print(f"> binning start: {os.path.basename(bam_path)}")

        # count read
        for read in samfile:
            if read.is_unmapped:
                continue
            idx = read.reference_start // bin_size
            chr = read.reference_name
            if chr in refs and idx < bin_nums[chr]:
                self.bin_profile_[chr].rd[idx] += 1
        samfile.close()

        # count N and gc
        for v in refs.values():
            v.stat(bin_size)

        # correct RD (del N, fill zero, correct gc bias)
        for chr, rd in self.bin_profile_.items():
            rd.rd = rd.rd[refs[chr].valid]
            rd.pos = rd.pos[refs[chr].valid]
            rd.rd /= bin_size
            rd.mode = mode_rd(rd.rd, bin_size)
            rd.rd[np.isclose(rd.rd, 0.0)] = rd.mode
            rd.rd = gc_correct(rd.rd, refs[chr].gc[refs[chr].valid])

            print(f"\tchrID: {chr}, binning num: {len(rd.rd)}, bin_size: {bin_size}")

        self.__bin_size = bin_size

    def read_fa(self, fa_path: dict):
        self.ref_profile_.clear()
        fa_path = {
            k: v
            for k, v in fa_path.items()
            if isinstance(k, str) and isinstance(v, str) and k in self.__chrs
        }
        for k, path in fa_path.items():
            fa_seq = SeqIO.read(path, "fasta")
            self.ref_profile_[k] = RDG.RefData(str(fa_seq.seq))
        self.ref_profile_ = {k: v for k, v in self.ref_profile_.items() if len(v) > 0}

    def __check_bam_ref(self, bam_path: str):
        samfile = pysam.AlignmentFile(bam_path, "rb", ignore_truncation=True)
        refs = {
            chr: self.ref_profile_[chr]
            for chr in samfile.references
            if chr in self.ref_profile_
        }
        if not refs:
            samfile.close()
            warnings.warn("No matching reference sequence index")
            return None

        return samfile, refs
