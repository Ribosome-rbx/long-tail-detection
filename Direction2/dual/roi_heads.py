# Copyright (c) Facebook, Inc. and its affiliates.
import pdb
import inspect
import logging
import copy
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn

from detectron2.layers import ShapeSpec, cat
from detectron2.config import configurable
from detectron2.structures import ImageList, Instances, pairwise_iou
from detectron2.modeling import StandardROIHeads, ROI_HEADS_REGISTRY
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.matcher import Matcher
from detectron2.structures import Boxes
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.utils.events import get_event_storage

from .contrastive_loss import build_contrastive_head, ContrastiveHead, SupConLoss
from .transformer import build_transformer
from .fast_rcnn import MemoryFastRCNNOutputLayers
from .lvis_v0_5_categories import get_image_count_frequency

logger = logging.getLogger(__name__)


@ROI_HEADS_REGISTRY.register()
class dualROIHeads(StandardROIHeads):
    @configurable
    def __init__(
        self,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_head: nn.Module,
        box_predictor: nn.Module,
        transformer: nn.Module,
        contrastive_head: nn.Module,
        mask_in_features: Optional[List[str]] = None,
        mask_pooler: Optional[ROIPooler] = None,
        mask_head: Optional[nn.Module] = None,
        keypoint_in_features: Optional[List[str]] = None,
        keypoint_pooler: Optional[ROIPooler] = None,
        keypoint_head: Optional[nn.Module] = None,
        train_on_pred_boxes: bool = False,
        contrastive_branch: bool = False,
        # memory bank parameters
        temp_S=None,
        min_cache=None,
        max_cache=None,
        cls_layer=None,
        random_select=None,
        cache_category_file=None,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            box_in_features (list[str]): list of feature names to use for the box head.
            box_pooler (ROIPooler): pooler to extra region features for box head
            box_head (nn.Module): transform features to make box predictions
            box_predictor (nn.Module): make box predictions from the feature.
                Should have the same interface as :class:`FastRCNNOutputLayers`.
            mask_in_features (list[str]): list of feature names to use for the mask
                pooler or mask head. None if not using mask head.
            mask_pooler (ROIPooler): pooler to extract region features from image features.
                The mask head will then take region features to make predictions.
                If None, the mask head will directly take the dict of image features
                defined by `mask_in_features`
            mask_head (nn.Module): transform features to make mask predictions
            keypoint_in_features, keypoint_pooler, keypoint_head: similar to ``mask_*``.
            train_on_pred_boxes (bool): whether to use proposal boxes or
                predicted boxes from the box head to train other heads.
        """
        super(StandardROIHeads, self).__init__(**kwargs)
        # keep self.in_features for backward compatibility
        self.in_features = self.box_in_features = box_in_features
        self.box_pooler = box_pooler
        self.box_head = box_head
        self.box_predictor = box_predictor
        self.transformer = transformer

        self.mask_on = mask_in_features is not None
        if self.mask_on:
            self.mask_in_features = mask_in_features
            self.mask_pooler = mask_pooler
            self.mask_head = mask_head

        self.contrastive_branch = contrastive_branch
        # Contrastive Loss head
        if self.contrastive_branch:
            self.contrastive_head = contrastive_head
        
        self.keypoint_on = keypoint_in_features is not None
        if self.keypoint_on:
            self.keypoint_in_features = keypoint_in_features
            self.keypoint_pooler = keypoint_pooler
            self.keypoint_head = keypoint_head

        self.train_on_pred_boxes = train_on_pred_boxes

        # memory bank
        self.cls_layer = cls_layer
        if "lvis" in cache_category_file:
            self.cache_categories = np.array(
                [int(x.rstrip()) - 1 for x in open(cache_category_file)]
            )
        else:
            logging.critical("not using lvis bank category file")

        if "lvis0" in cache_category_file:
            assert 1230 not in self.cache_categories
            assert self.num_classes == 1230
        elif "lvis1" in cache_category_file:
            assert 1203 not in self.cache_categories
            assert self.num_classes == 1203

        self.memory_cache = {
            c: {
                "box_features": np.empty((max_cache, 1024), dtype=np.float32),
                "proposals": np.empty((max_cache,), dtype=Instances),
            }
            for c in self.cache_categories
        }
        self.memory_cache_max_idx = np.zeros(self.num_classes, dtype=int)
        self.min_cache = min_cache
        self.max_cache = max_cache
        self.random_select = random_select
        
    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = {
            "batch_size_per_image": cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE,
            "positive_fraction": cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION,
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "proposal_append_gt": cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT,
            # Matcher to assign box proposals to gt boxes
            "proposal_matcher": Matcher(
                cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
                cfg.MODEL.ROI_HEADS.IOU_LABELS,
                allow_low_quality_matches=False,
            ),
            "train_on_pred_boxes": cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES,
            "contrastive_head": build_contrastive_head(cfg),
            "contrastive_branch": cfg.MODEL.ROI_HEADS.CONTRASTIVE_BRANCH,
            "transformer": build_transformer(cfg),
            # memory bank
            "cls_layer": cfg.MODEL.ROI_HEADS.CLS_LAYER,
            "temp_S": cfg.MODEL.ROI_HEADS.TEMP_S,
            "min_cache": cfg.MODEL.ROI_HEADS.MIN_CACHE,
            "max_cache": cfg.MODEL.ROI_HEADS.MAX_CACHE,
            "random_select": cfg.MODEL.ROI_HEADS.RANDOM_SELECT,
            "cache_category_file": cfg.MODEL.ROI_HEADS.CACHE_CAT_FILE,
        }
        # Subclasses that have not been updated to use from_config style construction
        # may have overridden _init_*_head methods. In this case, those overridden methods
        # will not be classmethods and we need to avoid trying to call them here.
        # We test for this with ismethod which only returns True for bound methods of cls.
        # Such subclasses will need to handle calling their overridden _init_*_head methods.
        if inspect.ismethod(cls._init_box_head):
            ret.update(cls._init_box_head(cfg, input_shape))
        if inspect.ismethod(cls._init_mask_head):
            ret.update(cls._init_mask_head(cfg, input_shape))
        if inspect.ismethod(cls._init_keypoint_head):
            ret.update(cls._init_keypoint_head(cfg, input_shape))
        return ret

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features              = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution        = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales            = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio           = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type              = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        box_head = build_box_head(
            cfg,
            ShapeSpec(
                channels=in_channels, height=pooler_resolution, width=pooler_resolution
            ),
        )
        box_predictor = MemoryFastRCNNOutputLayers(cfg, box_head.output_shape)

        freq_info = torch.FloatTensor(get_image_count_frequency())
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }

    def update_memory_bank(self, box_features, proposals):
        for p_idx in range(len(proposals)):
            p = proposals[p_idx]
            cur_gt_classes = p.gt_classes.cpu().numpy()
            rare_idxs = np.where(np.isin(cur_gt_classes, self.cache_categories))[0]
            if len(rare_idxs) > 0:
                for idx in rare_idxs:
                    c = cur_gt_classes[idx]
                    # append current feat to cache
                    feat = box_features[p_idx * len(p) + idx].detach().cpu().clone()
                    prop = copy.deepcopy(p[int(idx)])

                    # shift if exceeding max cache space
                    if self.memory_cache_max_idx[c] == self.max_cache:
                        self.memory_cache[c]["box_features"][
                            :-1, :
                        ] = self.memory_cache[c]["box_features"][1:, :]
                        self.memory_cache[c]["proposals"][:-1] = self.memory_cache[c][
                            "proposals"
                        ][1:]
                        # reset to last index (allow to grow by 1)
                        self.memory_cache_max_idx[c] = self.max_cache - 1
                    # append to cache
                    self.memory_cache[c]["box_features"][
                        self.memory_cache_max_idx[c]
                    ] = feat
                    self.memory_cache[c]["proposals"][
                        self.memory_cache_max_idx[c]
                    ] = prop

                    self.memory_cache_max_idx[c] += 1

    def use_memory_cache(self, box_features, proposals):
        # check if we're using memory bank
        if not self.min_cache:
            gt_classes = cat([p.gt_classes for p in proposals], dim=0)
            return proposals, box_features, gt_classes

        augmented_proposals = proposals

        # target classes that exist in the current batch
        target_classes = []
        for p_idx in range(len(proposals)):
            p = proposals[p_idx]
            cur_gt_classes = p.gt_classes.cpu().numpy()
            rare_idxs = np.where(np.isin(cur_gt_classes, self.cache_categories))[0]
            target_classes.extend(cur_gt_classes[rare_idxs])

        # count number of instances per category for any targeted category
        target_instances = dict()
        for i in target_classes:
            target_instances[i] = target_instances.get(i, 0) + 1

        new_proposals = []
        new_features = torch.tensor([]).cuda()
        for c in set(target_classes):
            num_samp_cache = self.memory_cache_max_idx[c]
            if num_samp_cache > 0:
                # get from cache x amount of samples. x = num_new_samps
                # use either the designated amount of samples (default 20) or the minimum amount in the current cache
                num_new_samps = min(self.min_cache, num_samp_cache)

                if self.random_select and num_samp_cache > num_new_samps:
                    cache_idxs = np.random.choice(
                        num_samp_cache, num_new_samps, replace=False
                    )
                else:
                    cache_idxs = np.arange(
                        num_samp_cache - num_new_samps, self.memory_cache_max_idx[c]
                    )

                new_feats = torch.from_numpy(
                    self.memory_cache[c]["box_features"][cache_idxs]
                )
                new_features = torch.cat((new_features, new_feats.cuda()), dim=0)
                if None in self.memory_cache[c]["proposals"][cache_idxs]:
                    pdb.set_trace()
                new_proposals.extend(self.memory_cache[c]["proposals"][cache_idxs])

        # update memory bank with current model
        self.update_memory_bank(box_features, proposals)

        if len(new_proposals) > 0:
            box_features = torch.cat((box_features, new_features), dim=0)
            augmented_proposals = copy.deepcopy(proposals)
            augmented_proposals.extend(new_proposals)

        all_gt_classes = cat([p.gt_classes for p in augmented_proposals], dim=0)
        assert len(all_gt_classes) == len(box_features)

        return augmented_proposals, box_features

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        rare_img_idx = None,
        cur_batchsize = None,
        rare_categories = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        if self.training:
            assert targets, "'targets' argument is required during training"
            # all_proposals = copy.deepcopy(proposals)
            proposals = self.label_and_sample_proposals(proposals, targets)
            # copy rare proposals for new features, proposals.dim(0) == images.dim(0)
            len_rare = 0
            if not cur_batchsize == len(proposals):
                proposals = self.copy_rare_proposals(
                    cur_batchsize, rare_img_idx, proposals, rare_categories)
                features, proposals, len_rare = self.filter_rare_proposals(
                    cur_batchsize, features, proposals, rare_categories)
        del targets, images

        if self.training:
            # losses = self._forward_box(features, proposals, len_rare)
            losses = self._forward_box(features, proposals, len_rare, rare_categories)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def _forward_box(self, features: Dict[str, torch.Tensor], proposals: List[Instances], len_rare = 0, rare_categories = None):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        if self.training:
            # memory bank
            augmented_proposals, box_features = self.use_memory_cache(
                box_features, proposals
            )
            if len(box_features) > 1024:
                breakpoint()
            # transformer
            query, key, augmented_proposals = self.get_qk(box_features, augmented_proposals, rare_categories)
            box_features = self.transformer(query, key, box_features)
        predictions = self.box_predictor(box_features)

        if self.training:
            losses = self.box_predictor.losses(predictions, augmented_proposals)
            if self.contrastive_branch:
                breakpoint()
                gt_classes = [prop.gt_classes for prop in augmented_proposals]
                gt_classes = torch.cat(gt_classes, dim=0)
                losses.update(self.contrastive_head(box_features, gt_classes, len_rare)) ###
            del box_features
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            del box_features
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances

    @torch.no_grad()   
    def get_qk(self, box_features, proposals, rare_categories): 
        # get query
        split_features = box_features
        query = []
        key = []
        att_proposals = []
        for prop in proposals:
            gt_classes = prop.gt_classes.cpu().numpy()
            rare_bool = np.isin(gt_classes, rare_categories)
            rare_idxs = np.where(rare_bool)[0]
            non_rare_idxs = np.where(~rare_bool)[0]
            key.append(split_features[non_rare_idxs])
            if not len(rare_idxs) == 0:
                query.append(split_features[rare_idxs])
                att_proposals.append(Instances.cat([prop[int(idx)] for idx in rare_idxs]))
            split_features = split_features[len(gt_classes):,]
        if not len(query) == 0:
            key = torch.cat(key, dim=0).unsqueeze(1)
            query = torch.cat(query, dim=0).unsqueeze(1)
            return query, key, proposals + att_proposals
        else:
            zero = torch.zeros_like(box_features[0]).unsqueeze(0).unsqueeze(0)
            return zero, zero, proposals

    @torch.no_grad()   
    def copy_rare_proposals(self, cur_batchsize, rare_img_idx, proposals, rare_categories):
        for i, p_idx in enumerate(rare_img_idx):
            p = proposals[p_idx]
            cur_gt_classes = p.gt_classes.cpu().numpy()
            rare_idxs = np.where(np.isin(cur_gt_classes, rare_categories))[0]
            if len(rare_idxs) > 0:
                rare_prop = [p[int(i)] for i in rare_idxs]
                # rare_prop.append(proposals[cur_batchsize + i])
                proposals[cur_batchsize + i] = Instances.cat(rare_prop)
        return proposals

    @torch.no_grad()
    def filter_rare_proposals(self, cur_batchsize, features, proposals, rare_categories):
        len_rare = 0
        feature_idx = [i for i in range(cur_batchsize)]
        rare_proposals = []
        for p_idx in range(cur_batchsize, len(proposals)):
            p = proposals[p_idx]
            cur_gt_classes = p.gt_classes.cpu().numpy()
            rare_idxs = np.where(np.isin(cur_gt_classes, rare_categories))[0]
            if len(rare_idxs) > 0:
                len_rare += len(rare_idxs)
                rare_prop = [p[int(i)] for i in rare_idxs]
                rare_proposals.append(Instances.cat(rare_prop))
                feature_idx.append(p_idx)
        proposals = proposals[:cur_batchsize] + rare_proposals
        for key in features.keys():
            features[key] = features[key][feature_idx]
        return features, proposals, len_rare


    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances]
    ) -> List[Instances]:
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        gt_boxes = [x.gt_boxes for x in targets]
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )

            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            
            # set all the gt proposals to be matched with the gt classes(prevent from leaking rare objects)
            M_classes = len(targets_per_image.gt_classes)
            matched_idxs[-M_classes:] = torch.tensor([i for i in range(M_classes)])

            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # We index all the attributes of targets that start with "gt_"
                # and have not been added to proposals yet (="gt_classes").
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            # If no GT is given in the image, we don't know what a dummy gt value can be.
            # Therefore the returned proposals won't have any gt_* fields, except for a
            # gt_classes full of background label.
            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)
        
        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt