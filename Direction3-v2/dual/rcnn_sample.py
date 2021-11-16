# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage

from detectron2.modeling import GeneralizedRCNN
from detectron2.modeling import Backbone, build_backbone
from detectron2.modeling import detector_postprocess
from detectron2.modeling import build_proposal_generator
from detectron2.modeling import build_roi_heads
from detectron2.modeling import META_ARCH_REGISTRY

__all__ = ["DualRCNN_sample"]


@META_ARCH_REGISTRY.register()
class DualRCNN_sample(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        momentum: float = 0.99,
        rare_category_file=None,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        self.avg_backbone = None
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"
        self.m = momentum

        # record rare classes
        if "lvis" in rare_category_file:
            self.rare_categories = np.array(
                [int(x.rstrip()) - 1 for x in open(rare_category_file)]
            )
        else:
            logging.critical("not using lvis bank category file")
        
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "momentum" : cfg.MODEL.BACKBONE.MOMENTUM,
            "rare_category_file" : cfg.MODEL.ROI_HEADS.RARE_CAT_FILE,
            ### Add a hyperparameter to control avg_backbone updating rate
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images, images_list = self.preprocess_image(batched_inputs) # images: type(ImageList), images_list: type(List)
        cur_batchsize = len(images_list)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        
        ### compute rare features
        with torch.no_grad():  # no gradient to avg-model
            if self.avg_backbone == None:
                self.avg_backbone = copy.deepcopy(self.backbone) # initialize the avg_backbone
                for avg_param in self.avg_backbone.parameters():
                    avg_param.requires_grad = False  # not update by gradient
            else:
                self._momentum_update_avg_backbone()  # update the avg_backbone
            rare_images_list, rare_gt_instances, rare_img_idx = self.extract_rare(images, gt_instances)
        
        # feed into avg_model to update batch
        if len(rare_images_list) > 0:
            rare_images = _from_tensors(images_list, rare_images_list, self.backbone.size_divisibility)
            rare_features = self.avg_backbone(rare_images.tensor)

            # concatenate origional batch with rare_image set
            new_images_list = images_list + rare_images_list
            new_images = ImageList.from_tensors(new_images_list, self.backbone.size_divisibility)
            
            # concatenate origional gt with rare_image gt
            new_gt_instances = gt_instances + rare_gt_instances 
            # concatenate origional feature with new features
            new_features = {}
            for key in features.keys():
                new_features[key] = torch.cat([features[key], rare_features[key]], dim=0)
        else:
            new_images = images
            new_gt_instances = gt_instances
            new_features = features

        if self.proposal_generator is not None:
            # only origional image batch are used to generate proposals
            proposals, proposal_losses = self.proposal_generator(new_images, new_features, new_gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}
        
        # both origional batch and rare images are fed into roi_heads, where we expand the proposals for rare images
        _, detector_losses = self.roi_heads(new_images, new_features, proposals, new_gt_instances, rare_img_idx, cur_batchsize, self.rare_categories)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    @torch.no_grad()
    def _momentum_update_avg_backbone(self):
        """
        Momentum update of the avg_backbone
        """
        for param, avg_param in zip(self.backbone.parameters(), self.avg_backbone.parameters()):
            avg_param.data = avg_param.data * self.m + param.data * (1. - self.m)

    @torch.no_grad()
    def extract_rare(self, images, gt_instances):
        #extract rare image from batch
        rare_images_list = []
        rare_gt_instances = []
        rare_img_idx = []
        for image, instances in zip(images, gt_instances):
            if True in np.isin(instances.gt_classes.to('cpu'), self.rare_categories):
                rare_images_list.append(image)
                rare_gt_instances.append(instances)
                rare_img_idx.append(gt_instances.index(instances))
        return rare_images_list, rare_gt_instances, rare_img_idx

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images, _ = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images_list = [x["image"].to(self.device) for x in batched_inputs]
        images_list = [(x - self.pixel_mean) / self.pixel_std for x in images_list]
        images = ImageList.from_tensors(images_list, self.backbone.size_divisibility)
        return images, images_list

    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

def _from_tensors(images: List[torch.Tensor], rare_images: List[torch.Tensor], size_divisibility: int = 0, pad_value: float = 0.0):
        assert len(rare_images) > 0
        assert isinstance(rare_images, (tuple, list))
        for t in rare_images:
            assert isinstance(t, torch.Tensor), type(t)
            assert t.shape[:-2] == rare_images[0].shape[:-2], t.shape
        # compute max_size of this batch
        image_sizes = [(im.shape[-2], im.shape[-1]) for im in images]
        image_sizes_tensor = [torch.as_tensor(x) for x in image_sizes]
        max_size = torch.stack(image_sizes_tensor).max(0).values
        if size_divisibility > 1:
            stride = size_divisibility
            # the last two dims are H,W, both subject to divisibility requirement
            max_size = (max_size + (stride - 1)) // stride * stride

        # store rare image_sizes
        image_sizes = [(im.shape[-2], im.shape[-1]) for im in rare_images]
        image_sizes_tensor = [torch.as_tensor(x) for x in image_sizes]

        # handle weirdness of scripting and tracing ...
        if torch.jit.is_scripting():
            max_size: List[int] = max_size.to(dtype=torch.long).tolist()
        else:
            if torch.jit.is_tracing():
                image_sizes = image_sizes_tensor

        if len(rare_images) == 1:
            # This seems slightly (2%) faster.
            # TODO: check whether it's faster for multiple images as well
            image_size = image_sizes[0]
            padding_size = [0, max_size[-1] - image_size[1], 0, max_size[-2] - image_size[0]]
            batched_imgs = F.pad(rare_images[0], padding_size, value=pad_value).unsqueeze_(0)
        else:
            # max_size can be a tensor in tracing mode, therefore convert to list
            batch_shape = [len(rare_images)] + list(rare_images[0].shape[:-2]) + list(max_size)
            batched_imgs = rare_images[0].new_full(batch_shape, pad_value)
            for img, pad_img in zip(rare_images, batched_imgs):
                pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)
        return ImageList(batched_imgs.contiguous(), image_sizes)
