from easydict import EasyDict

import logging
import os
import typing as t

cfg = EasyDict()

cfg.DATASET_DESCRIPTION = os.path.join("configs", "dataset", "shapenet.json")
cfg.CLASS_MAPPING = os.path.join("configs", "dataset", "class_mapping.json")
cfg.CLASS_TO_INDEX_MAPPING = os.path.join(
    "configs", "dataset", "class_to_index_mapping.json"
)
cfg.TRAIN_OBJECT_SPLIT_CONFIG = os.path.join(
    "configs", "dataset", "train_object_split.json"
)
cfg.VALID_OBJECT_SPLIT_CONFIG = os.path.join(
    "configs", "dataset", "valid_object_split.json"
)

cfg.TRAIN_SYNSET_SPLIT_CONFIG = os.path.join(
    "configs", "dataset", "train_synset_split.json"
)
cfg.VALID_SYNSET_SPLIT_CONFIG = os.path.join(
    "configs", "dataset", "valid_synset_split.json"
)

cfg.DIR = EasyDict()
cfg.DIR.DATASET = os.path.join(
    "/datasets", "users", "kkania", "shapenet", "raw"
)

cfg.DIR.PREPROCESSED_DATASET = os.path.join(
    "/datasets", "users", "kkania", "shapenet", "preprocessed"
)

cfg.CONST = EasyDict()
cfg.CONST.RNG_SEED = 1337
cfg.CONST.IMG_H = 128
cfg.CONST.IMG_W = 128
cfg.CONST.BATCH_SIZE = 64

cfg.CONST.TRAIN_VALID_OBJECTS_RATIO = 0.7
cfg.CONST.TRAIN_VALID_SYNSETS_RATIO = 0.7

cfg.CONST.ENCODER_FILE_BASE_NAME = "encoder"
cfg.CONST.DECODER_FILE_BASE_NAME = "decoder"
cfg.CONST.OPTIMIZER_FILE_BASE_NAME = "optimizer"
cfg.CONST.HISTORY_FILE_BASE_NAME = "optimizer"

ConfigType = t.Dict[str, t.Dict[str, t.Any]]

logging.basicConfig(level=logging.INFO)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    handler = logging.FileHandler("logs/log.log", mode="a")
    handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(stream_handler)

    return logger
