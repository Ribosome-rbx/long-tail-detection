def add_config(cfg):
    cfg.MODEL.BACKBONE.MOMENTUM = 0.99
    cfg.MODEL.ROI_HEADS.RARE_CAT_FILE = "lvis0.5_rare_cats.txt"
    # transformer configs
    cfg.MODEL.BACKBONE.NUM_HEADS = 8
    cfg.MODEL.BACKBONE.NUM_DECODER_LAYERS = 6
    cfg.MODEL.BACKBONE.DIM_FEEDFORWARD = 2048
    cfg.MODEL.BACKBONE.DROPOUT = 0.1
    cfg.MODEL.BACKBONE.ACTIVATION = "relu"
    cfg.MODEL.BACKBONE.NORMALIZE_BEFORE = False
    cfg.MODEL.BACKBONE.NORM = None
    cfg.MODEL.BACKBONE.RETURN_INTERMEDIATE = False
    cfg.MODEL.BACKBONE.TRANSFORMER_WEIGHT = 0.1
    # contrastive configs
    cfg.MODEL.ROI_HEADS.CONTRASTIVE_BRANCH = False
    cfg.MODEL.ROI_BOX_HEAD.MLP_FEATURE_DIM = 128
    cfg.MODEL.ROI_BOX_HEAD.TEMPERATURE = 0.1
    cfg.MODEL.ROI_BOX_HEAD.LOSS_WEIGHT = 1.0
