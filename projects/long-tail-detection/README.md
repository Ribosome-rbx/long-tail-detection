# Illustrations on Source Code
## [configs](https://github.com/Ribosome-rbx/long-tail-detection/tree/main/projects/long-tail-detection/configs)
- Dual-RCNN-sample.yaml: used for final model training and evaluating.
- Dual-RCNN-con.yaml: use contrastive learning on top of final model
- 
## [project_models](https://github.com/Ribosome-rbx/long-tail-detection/tree/main/projects/long-tail-detection/project_models)
Files used for building models
- rcnn_sample.py: build momentum updated branch (all relative codes are annotated)
- roi_heads.py: build the memory bank
- transformer.py: build the Transformer block
- contrastive_loss.py: build the contrastive branch

## [lvis_categories_lists](https://github.com/Ribosome-rbx/long-tail-detection/tree/main/projects/long-tail-detection/lvis_categories_lists)
TXT files which divide LVIS classes in groups
