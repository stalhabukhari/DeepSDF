import os

from easydict import EasyDict

cfg = EasyDict()

cfg.DATASET_DESCRIPTION = os.path.join(
    "experiments", "dataset", "shapenet.json"
)
cfg.TRAIN_OBJECT_SPLIT_CONFIG = os.path.join(
    "experiments", "dataset", "train_object_split.json"
)
cfg.VALID_OBJECT_SPLIT_CONFIG = os.path.join(
    "experiments", "dataset", "valid_object_split.json"
)

cfg.TRAIN_SYNSET_SPLIT_CONFIG = os.path.join(
    "experiments", "dataset", "train_synset_split.json"
)
cfg.VALID_SYNSET_SPLIT_CONFIG = os.path.join(
    "experiments", "dataset", "valid_synset_split.json"
)

cfg.DIR = EasyDict()
cfg.DIR.DATASET = os.path.join(
    "/datasets", "users", "kkania", "shapenet", "ShapeNetCore.v2"
)

cfg.CONST = EasyDict()
cfg.CONST.RNG_SEED = 1337
cfg.CONST.IMG_H = 127
cfg.CONST.IMG_W = 127
cfg.CONST.BATCH_SIZE = 64

cfg.CONST.TRAIN_VALID_OBJECTS_RATIO = 0.7
cfg.CONST.TRAIN_VALID_SYNSETS_RATIO = 0.7
