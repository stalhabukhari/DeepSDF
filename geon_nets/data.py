#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import glob
import json
import logging
import os
import random
import typing as t
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.utils.data
import tqdm
from functional import seq

import geon_nets.workspace as ws
from common import cfg


def get_instance_filenames(data_source, split):
    npzfiles = []
    for dataset in split:
        for class_name in split[dataset]:
            for instance_name in split[dataset][class_name]:
                instance_filename = os.path.join(
                    dataset, class_name, instance_name, "samples.npz"
                )
                if not os.path.isfile(
                    os.path.join(
                        data_source, ws.sdf_samples_subdir, instance_filename
                    )
                ):
                    logging.warning(
                        "Requested non-existent file '{}'".format(
                            instance_filename
                        )
                    )
                    continue
                npzfiles += [instance_filename]
    return npzfiles


class NoMeshFileError(RuntimeError):
    """Raised when a mesh file is not found in a shape directory"""


class MultipleMeshFileError(RuntimeError):
    """"Raised when a there a multiple mesh files in a shape directory"""


def find_mesh_in_directory(shape_dir):
    mesh_filenames = list(glob.iglob(shape_dir + "/**/*.obj")) + list(
        glob.iglob(shape_dir + "/*.obj")
    )
    if len(mesh_filenames) == 0:
        raise NoMeshFileError()
    elif len(mesh_filenames) > 1:
        raise MultipleMeshFileError()
    return mesh_filenames[0]


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]


def read_sdf_samples_into_ram(filename):
    npz = np.load(filename)
    pos_tensor = torch.from_numpy(npz["pos"])
    neg_tensor = torch.from_numpy(npz["neg"])

    return [pos_tensor, neg_tensor]


def unpack_sdf_samples(filename, subsample=None):
    npz = np.load(filename)
    if subsample is None:
        return npz
    pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
    neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))

    # split the sample into half
    half = int(subsample / 2)

    random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
    random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    samples = torch.cat([sample_pos, sample_neg], 0)

    return samples


def unpack_sdf_samples_from_ram(data, subsample=None):
    if subsample is None:
        return data
    pos_tensor = data[0]
    neg_tensor = data[1]

    # split the sample into half
    half = int(subsample / 2)

    pos_size = pos_tensor.shape[0]
    neg_size = neg_tensor.shape[0]

    pos_start_ind = random.randint(0, pos_size - half)
    sample_pos = pos_tensor[pos_start_ind : (pos_start_ind + half)]

    if neg_size <= half:
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)
    else:
        neg_start_ind = random.randint(0, neg_size - half)
        sample_neg = neg_tensor[neg_start_ind : (neg_start_ind + half)]

    samples = torch.cat([sample_pos, sample_neg], 0)

    return samples


class SDFSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample,
        load_ram=False,
        print_filename=False,
        num_files=1000000,
    ):
        self.subsample = subsample

        self.data_source = data_source
        self.npyfiles = get_instance_filenames(data_source, split)

        logging.debug(
            "using "
            + str(len(self.npyfiles))
            + " shapes from data source "
            + data_source
        )

        self.load_ram = load_ram

        if load_ram:
            self.loaded_data = []
            for f in self.npyfiles:
                filename = os.path.join(
                    self.data_source, ws.sdf_samples_subdir, f
                )
                npz = np.load(filename)
                pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
                neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))
                self.loaded_data.append(
                    [
                        pos_tensor[torch.randperm(pos_tensor.shape[0])],
                        neg_tensor[torch.randperm(neg_tensor.shape[0])],
                    ]
                )

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):
        filename = os.path.join(
            self.data_source, ws.sdf_samples_subdir, self.npyfiles[idx]
        )
        if self.load_ram:
            return (
                unpack_sdf_samples_from_ram(
                    self.loaded_data[idx], self.subsample
                ),
                idx,
            )
        else:
            return unpack_sdf_samples(filename, self.subsample), idx


class ImageToSDFDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split_file: str,
        class_mapping: str,
        subsample: t.Optional[int] = 1_024,
        verbose: bool = True,
        is_test: bool = False,
    ):
        self.split_file = split_file
        self.subsample = subsample
        self.class_mapping = json.loads(Path(class_mapping).read_text())
        self.split = json.loads(Path(split_file).read_text())
        self.samples = self._preprocess_split()
        self.verbose = verbose
        self.is_test = is_test

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> t.Tuple[torch.Tensor, ...]:
        (img_path, sdf_samples_path, _, class_name) = self.samples[idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (cfg.CONST.IMG_W, cfg.CONST.IMG_H))

        img = torch.from_numpy(img).float() / 255.0
        sdf_samples = unpack_sdf_samples(
            sdf_samples_path, subsample=self.subsample
        )

        class_num = torch.IntTensor([self.class_mapping[class_name]])
        coordinates = sdf_samples[:, -1]
        sdfs = sdf_samples[:, -1:]

        return img, coordinates, sdfs, class_num

    def _preprocess_split(self) -> t.List[t.Tuple]:
        samples = []
        for a_class, objects in tqdm.tqdm(self.split.items()):
            for _, paths in objects.items():
                renders = paths["renders"]
                sdf_samples = paths["sdf_samples"]
                surface_samples = paths["surface_samples"]

                samples.extend(
                    seq(renders)
                    .map(
                        lambda a_path: tuple(
                            (a_path, sdf_samples, surface_samples, a_class)
                        )
                    )
                    .to_list()
                )
        return samples


if __name__ == "__main__":
    from common import cfg

    dataset = ImageToSDFDataset(
        cfg.TRAIN_OBJECT_SPLIT_CONFIG, cfg.CLASS_TO_INDEX_MAPPING, 1_024
    )
    dataset[0]
