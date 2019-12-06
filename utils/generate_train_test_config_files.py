"""
Split of categories is based on Choy et al.
https://arxiv.org/pdf/1604.00449.pdf
"""
import json
import typing as t
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
from functional import seq

from common import ConfigType, cfg


def generate_config_and_name_mapping(
    input_directory: str, dataset_description_path: str
) -> t.Tuple[ConfigType, t.Dict[str, str]]:
    with open(dataset_description_path) as f:
        description = json.load(f)

    description = list(description.values())

    input_directory = Path(input_directory)
    output_config = {}
    name_mapping = {}

    non_normalized_count = 0
    all_shapes_count = 0
    non_normalized_shape_names = []

    for synset in description:
        an_id = synset["id"]
        class_name = synset["name"]
        partial_config = defaultdict(dict)

        for an_object_folder in (input_directory / an_id).glob("*"):
            png_synset_paths = list(an_object_folder.rglob("*.png"))
            rendering_paths = seq(png_synset_paths).filter(
                lambda a_path: a_path.parent.name == "rendering"
            )

            if rendering_paths.size() == 0:
                print("No renders at:", an_object_folder.absolute().as_posix())
                continue

            normalized_model_path = (
                rendering_paths[0].parent.parent
                / "models"
                / "model_normalized.obj"
            )

            if not normalized_model_path.exists():
                non_normalized_count += 1
                non_normalized_shape_names.append(
                    normalized_model_path.parent.parent.parent.name
                )
                continue
            all_shapes_count += 1

            partial_config[an_object_folder.name][
                "renders"
            ] = rendering_paths.map(lambda a_path: a_path.as_posix()).to_list()

            partial_config[an_object_folder.name][
                "model_path"
            ] = normalized_model_path.as_posix()

        output_config[an_id] = partial_config
        name_mapping[an_id] = class_name

    print(
        f"Non normalized to all shapes ratio: "
        f"{non_normalized_count / all_shapes_count:.4f}"
    )
    print(
        f"Non normalized and all shapes counts: "
        f"{non_normalized_count}, {all_shapes_count}"
    )
    print(
        f"Names of non normalized shapes: "
        f"{list(set(non_normalized_shape_names))}"
    )
    print(
        f"Names of non normalized counts: "
        f"{Counter(non_normalized_shape_names)}"
    )
    return output_config, name_mapping


def config_to_split_by_object_id(
    config: ConfigType,
) -> t.Tuple[ConfigType, ConfigType]:
    train_config = defaultdict(dict)
    valid_config = defaultdict(dict)

    rng = np.random.RandomState(cfg.CONST.RNG_SEED)

    for synset_name, synset_data in config.items():
        object_names = np.asarray(list(synset_data.keys()))
        indices = np.arange(0, len(object_names))
        rng.shuffle(indices)

        split_index = int(len(indices) * cfg.CONST.TRAIN_VALID_OBJECTS_RATIO)
        train_indices = indices[:split_index]
        valid_indices = indices[split_index:]

        train_object_names = object_names[train_indices]
        valid_object_names = object_names[valid_indices]

        for object_name in train_object_names:
            train_config[synset_name][object_name] = synset_data[object_name]

        for object_name in valid_object_names:
            valid_config[synset_name][object_name] = synset_data[object_name]

    return train_config, valid_config


def config_to_split_by_synset(
    config: ConfigType,
) -> t.Tuple[ConfigType, ConfigType]:
    train_config = defaultdict(dict)
    valid_config = defaultdict(dict)

    rng = np.random.RandomState(cfg.CONST.RNG_SEED)
    synset_names = np.asarray(list(config.keys()))
    indices = np.arange(0, len(synset_names))
    rng.shuffle(indices)

    split_index = int(len(indices) * cfg.CONST.TRAIN_VALID_SYNSETS_RATIO)
    train_indices = indices[:split_index]
    valid_indices = indices[split_index:]

    train_synsets_names = synset_names[train_indices]
    valid_synsets_names = synset_names[valid_indices]

    for synset_name in train_synsets_names:
        train_config[synset_name] = config[synset_name]

    for synset_name in valid_synsets_names:
        valid_config[synset_name] = config[synset_name]

    return train_config, valid_config


def dump_to_file(path: str, config: t.Dict[t.Any, t.Any]) -> None:
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_directory", default=cfg.DIR.DATASET)
    parser.add_argument("-c", "--config_path", default=cfg.DATASET_DESCRIPTION)
    parser.add_argument(
        "--train_object_output", default=cfg.TRAIN_OBJECT_SPLIT_CONFIG
    )
    parser.add_argument(
        "--valid_object_output", default=cfg.VALID_OBJECT_SPLIT_CONFIG
    )
    parser.add_argument(
        "--train_synset_output", default=cfg.TRAIN_SYNSET_SPLIT_CONFIG
    )
    parser.add_argument(
        "--valid_synset_output", default=cfg.VALID_SYNSET_SPLIT_CONFIG
    )
    parser.add_argument("--split_by_object", default=True, action="store_true")
    parser.add_argument("--split_by_synset", default=True, action="store_true")

    args = parser.parse_args()

    a_config, class_mapping = generate_config_and_name_mapping(
        args.input_directory, args.config_path
    )

    dump_to_file(cfg.CLASS_MAPPING, class_mapping)

    if args.split_by_object:
        splits = config_to_split_by_object_id(a_config)
        dump_to_file(args.train_object_output, splits[0])
        dump_to_file(args.valid_object_output, splits[1])

    if args.split_by_synset:
        splits = config_to_split_by_synset(a_config)
        dump_to_file(args.train_synset_output, splits[0])
        dump_to_file(args.valid_synset_output, splits[1])


if __name__ == "__main__":
    main()
