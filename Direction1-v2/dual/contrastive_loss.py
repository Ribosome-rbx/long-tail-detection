import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init

def build_contrastive_head(cfg):
    fc_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
    out_dim = cfg.MODEL.ROI_BOX_HEAD.MLP_FEATURE_DIM
    temperature = cfg.MODEL.ROI_BOX_HEAD.TEMPERATURE
    contrast_loss_weight = cfg.MODEL.ROI_BOX_HEAD.LOSS_WEIGHT
    return ContrastiveHead(fc_dim, out_dim, temperature, contrast_loss_weight)

class ContrastiveHead(nn.Module):
    """MLP head for contrastive representation learning, https://arxiv.org/abs/2003.04297
    Args:
        dim_in (int): dimension of the feature intended to be contrastively learned
        feat_dim (int): dim of the feature to calculated contrastive loss

    Return:
        feat_normalized (tensor): L-2 normalized encoded feature,
            so the cross-feature dot-product is cosine similarity (https://arxiv.org/abs/2004.11362)
    """
    def __init__(self, dim_in, feat_dim, temperature, contrast_loss_weight):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, feat_dim),
        )
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                weight_init.c2_xavier_fill(layer)
        self.criterion1 = SupConLoss(temperature)
        self.criterion2 = SupConLoss2(temperature)
        self.contrast_loss_weight = contrast_loss_weight

    def forward(self, box_features, gt_classes, len_rare):
        # if len_rare == 0:
        #     feat = self.head(box_features)
        #     feat_normalized = F.normalize(feat, dim=1)
        #     contrastive_loss = self.criterion1(feat_normalized, gt_classes)
        # else:
        #     x = box_features[-len_rare:,:]
        #     x_gt_classes = gt_classes[-len_rare:]
        #     y = box_features[:-len_rare,:]
        #     y_gt_classes = gt_classes[:-len_rare]
            
        #     x_feat = self.head(x)
        #     x_feat_normalized = F.normalize(x_feat, dim=1)
        #     y_feat = self.head(y)
        #     y_feat_normalized = F.normalize(y_feat, dim=1)
        #     contrastive_loss = self.criterion2(x_feat_normalized, x_gt_classes, y_feat_normalized, y_gt_classes)
        feat = self.head(box_features)
        feat_normalized = F.normalize(feat, dim=1)
        contrastive_loss = self.criterion1(feat_normalized, gt_classes)
        return {'loss_contrast': self.contrast_loss_weight * contrastive_loss}


class SupConLoss2(nn.Module):
    """Supervised Contrastive LOSS as defined in https://arxiv.org/pdf/2004.11362.pdf."""

    def __init__(self, temperature=0.2, iou_threshold=0.5, reweight_func='none'):
        '''Args:
            tempearture: a constant to be divided by consine similarity to enlarge the magnitude
            iou_threshold: consider proposals with higher credibility to increase consistency.
        '''
        super().__init__()
        self.temperature = temperature
        self.iou_threshold = iou_threshold
        self.reweight_func = reweight_func

    def forward(self, x_features, x_labels, y_features, y_labels):
        """
        Args:
            features (tensor): shape of [M, K] where M is the number of features to be compared,
                and K is the feature_dim.   e.g., [8192, 128]
            labels (tensor): shape of [M].  e.g., [8192]
        """
        assert x_features.shape[0] == x_labels.shape[0]
        assert y_features.shape[0] == y_labels.shape[0]

        if len(x_labels.shape) == 1 or len(y_labels.shape):
            x_labels = x_labels.reshape(-1, 1) # label: [1024,] -> [1024, 1]
            y_labels = y_labels.reshape(-1, 1) # label: [1024,] -> [1024, 1]

        # mask of shape [None, None], mask_{i, j}=1 if sample i and sample j have the same label
        label_mask = torch.eq(x_labels, y_labels.T).float().cuda()
        # label_mask: [1024, 1024]
        # features: [1024, 128]
        similarity = torch.div(
            torch.matmul(x_features, y_features.T), self.temperature)
        # similarity: [1024, 1024]
        # for numerical stability
        sim_row_max, _ = torch.max(similarity, dim=1, keepdim=True)
        # sim_row_max: [1024, 1]
        similarity = similarity - sim_row_max.detach()

        # # mask out self-contrastive
        # logits_mask = torch.ones_like(similarity)
        # logits_mask.fill_diagonal_(0)

        exp_sim = torch.exp(similarity)
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))
        if 0 in label_mask.sum(1):
            breakpoint()
        per_label_log_prob = (log_prob * label_mask).sum(1) / label_mask.sum(1)

        loss = -per_label_log_prob
        return loss.mean()

    @staticmethod
    def _get_reweight_func(option):
        def trivial(iou):
            return torch.ones_like(iou)
        def exp_decay(iou):
            return torch.exp(iou) - 1
        def linear(iou):
            return iou

        if option == 'none':
            return trivial
        elif option == 'linear':
            return linear
        elif option == 'exp':
            return exp_decay

class SupConLoss(nn.Module):
    """Supervised Contrastive LOSS as defined in https://arxiv.org/pdf/2004.11362.pdf."""

    def __init__(self, temperature=0.2, iou_threshold=0.5, reweight_func='none'):
        '''Args:
            tempearture: a constant to be divided by consine similarity to enlarge the magnitude
            iou_threshold: consider proposals with higher credibility to increase consistency.
        '''
        super().__init__()
        self.temperature = temperature
        self.iou_threshold = iou_threshold
        self.reweight_func = reweight_func

    def forward(self, features, labels):
        """
        Args:
            features (tensor): shape of [M, K] where M is the number of features to be compared,
                and K is the feature_dim.   e.g., [8192, 128]
            labels (tensor): shape of [M].  e.g., [8192]
        """
        assert features.shape[0] == labels.shape[0]

        if len(labels.shape) == 1:
            labels = labels.reshape(-1, 1)

        # mask of shape [None, None], mask_{i, j}=1 if sample i and sample j have the same label
        label_mask = torch.eq(labels, labels.T).float().cuda()

        similarity = torch.div(
            torch.matmul(features, features.T), self.temperature)
        # for numerical stability
        sim_row_max, _ = torch.max(similarity, dim=1, keepdim=True)
        similarity = similarity - sim_row_max.detach()

        # mask out self-contrastive
        logits_mask = torch.ones_like(similarity)
        logits_mask.fill_diagonal_(0)

        exp_sim = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))

        per_label_log_prob = (log_prob * logits_mask * label_mask).sum(1) / label_mask.sum(1)
        loss = -per_label_log_prob
        return loss.mean()

    @staticmethod
    def _get_reweight_func(option):
        def trivial(iou):
            return torch.ones_like(iou)
        def exp_decay(iou):
            return torch.exp(iou) - 1
        def linear(iou):
            return iou

        if option == 'none':
            return trivial
        elif option == 'linear':
            return linear
        elif option == 'exp':
            return exp_decay
