# Illustration on different directions
## Direction1:
Resample rare imgs with avg_backbone
- Direction1-v2: the base (named "d1" in the doc)
- Direction1-v3: d1 with losses splited into cur_loss and avg_loss for the origional batch and the resmapled images

## Direction2:
Transformer from head features to tail features (no avg_backbone is used)
- in "master" branch: the base (named "d2" in the doc)
- in "memorybank" branch: built on top of RIO (named "RIO+d2" in the doc)

## Direction3:
Transformer from avg_rare to cur_rare images (with avg_backbone, built on top of "d1")

## Direction4:
A failure trial on modifying contrastive learning scheme.

## Direction5:
Apply transformer on the output of memory bank (for details, view the sildes on 26th Nov)
*Currently developing
