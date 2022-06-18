# Long-tail Object Detection

## Abstract
We built the averaged backbone, Transformer block and contrastive branch, and check their impact on detection with experiments. Our final model leverages the memory bank to collect rare samples and to resample randomly; generate new rare samples with the attention mechanism of Transformer block; distinguish different foreground classes by contrastive learning and trained in the multi-tasks fashion. The final model trained on LVIS dataset, surpasses other models, gains near 3 percent improvement on mAP compared with the Backbone. In the end, we analyze the shortcomings of current model and provide future directions. <br>

## Installation

#### Environment
- Python >= 3.6
- PyTorch 1.6.0 with CUDA 10.2 --> Refer to download guildlines at the [PyTorch website](pytorch.org)
- [Detectron2 v0.4](https://github.com/facebookresearch/detectron2/releases/tag/v0.4)
- OpenCV is optional but required for visualizations

#### Detectron2 
Please refer to the installation instructions in [Detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).<br>

#### LVIS Dataset 
Dataset download is available at the official [LVIS website](https://www.lvisdataset.org/dataset). Please follow [Detectron's guildlines](https://github.com/facebookresearch/detectron2/tree/master/datasets) on expected LVIS dataset structure.<br>

Install lvis-api by:
```
pip install git+https://github.com/lvis-dataset/lvis-api.git
```

## Training & Evaluation

Our code is located under [projects/long-tail-detection](https://github.com/Ribosome-rbx/long-tail-detection/tree/main/projects/long-tail-detection). <br>

Our training and evaluation follows those of Detectron2's. The config files for both LVISv0.5 and LVISv1.0 are provided.

Example: Training LVISv0.5 on Mask-RCNN ResNet-50
- For multi-gpu training (advised)
```
cd projects/long-tail-detection
python dual_train_net.py \
--num-gpus 4 \
--config-file ./configs/Dual-RCNN-sample.yaml OUTPUT_DIR ./outputs
```
- For single-gpu training, we need to adjust the learning rate and batchsize
```
python dual_train_net.py \
--num-gpus 1 \
--config-file ./configs/Dual-RCNN-sample.yaml \
SOLVER.BASE_LR 0.0025 SOLVER.IMS_PER_BATCH 2 OUTPUT_DIR ./outputs
```

Example: Evaluating LVISv0.5 on Mask-RCNN ResNet-50
```
cd projects/long-tail-detection
python dual_train_net.py \
--eval-only MODEL.WEIGHTS /path/to/model_checkpoint \
--config-file ./configs/Dual-RCNN-sample.yaml OUTPUT_DIR ./outputs
```

By default, LVIS evaluation follows immediately after training. 

## Ablation Study
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Memory<br/>Bank</th>
<th valign="bottom">Transformer</th>
<th valign="bottom">Contrastive<br/>Branch</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">box<br/>AP.r</th>
<th valign="bottom">box<br/>AP.c</th>
<th valign="bottom">box<br/>AP.f</th>
<th valign="bottom">mask<br/>AP</th>

<!-- TABLE BODY -->
<!-- ROW: 1 -->
 <tr><td align="center">✓</td>
<td align="center"></td>
<td align="center"></td>
<td align="center">22.035</td>
<td align="center">16.573</td>
<td align="center">19.456</td>
<td align="center">27.445</td>
<td align="center">22.606</td>
</tr>
<!-- ROW: 2 -->
 <tr><td align="center">[]</td>
<td align="center">[✓]</td>
<td align="center">[]</td>
<td align="center">21.860</td>
<td align="center">11.003</td>
<td align="center">20.673</td>
<td align="center">27.682</td>
<td align="center">22.663</td>
</tr>
<!-- ROW: 3 -->
 <tr><td align="center">[✓]</td>
<td align="center">[✓]</td>
<td align="center">[]</td>
<td align="center">23.029</td>
<td align="center">14.389</td>
<td align="center">21.793</td>
<td align="center">28.028</td>
<td align="center">23.399</td>
</tr>
<!-- ROW: 4 -->
 <tr><td align="center">[✓]</td>
<td align="center">[]</td>
<td align="center">[✓]</td>
<td align="center">22.015</td>
<td align="center">**18.457**</td>
<td align="center">18.873</td>
<td align="center">27.371</td>
<td align="center">22.536</td>
</tr>
<!-- ROW: 5 -->
 <tr><td align="center">[✓]</td>
<td align="center">[✓]</td>
<td align="center">[✓]</td>
<td align="center">**23.551**</td>
<td align="center">15.454</td>
<td align="center">**22.532**</td>
<td align="center">**28.060**</td>
<td align="center">**23.935**</td>
</tr>
</tbody></table>

## Visualization
Detectron2 has built-in visualization tools. Under tools folder, visualize_json_results.py can be used to visualize the json instance detection/segmentation results given by LVISEvaluator. 

```
python visualize_json_results.py --input x.json --output dir/ --dataset lvis
```

Further information can be found on [Detectron2 tools' README](https://github.com/facebookresearch/detectron2/tree/master/tools).
