# CNV-box
Some demo code about CNV detection.


## Model

### Class RDG

> A RD profile generator.

```python
# usage

# chr is a chromosome name list
rdg = RDG(chr=['chr21'])

# 1. direct preprocess
# chr is a chromosome name string, must in initial chr list
# Return RD profile after segment
(rd, start, end), mode = rdg.preprocess(bam, fa, chr='chr21')

# Too lazy to write ...
```

### cnvutils

> Some util function.

+ `read_gt`: Read groundtruth to special format.
+ `calc_result`: Calculate precesion and sensitivity.
+ `draw_profile`: Plot segment RD profile and groundtruth area.

### segutils

> Some segment strategy.

+ `cbs_seg`: CBS algorithm.
+ `no_seg`: bin as seg.