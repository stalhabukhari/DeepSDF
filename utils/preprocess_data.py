#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import concurrent.futures
import json
import logging
import multiprocessing as mp
import subprocess
import typing as t
from collections import Counter
from pathlib import Path

import tqdm

import geon_nets
import geon_nets.workspace as ws
from common import cfg


def process_mesh(
    mesh_filepath: str,
    target_filepath: str,
    executable: str,
    additional_args: t.List[str],
):
    command = [
        executable,
        "-m",
        mesh_filepath,
        "-o",
        target_filepath,
    ] + additional_args

    subproc = subprocess.Popen(command, stdout=subprocess.DEVNULL)
    subproc.wait()


def main():
    import argparse

    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Pre-processes data from a data source and append "
        "the results to a dataset.",
    )
    arg_parser.add_argument(
        "--source_dir",
        "-s",
        dest="source_dir",
        help="The directory which holds all raw data.",
        default=cfg.DIR.DATASET,
        type=Path,
    )
    arg_parser.add_argument(
        "--data_dir",
        "-d",
        dest="data_dir",
        help="The directory which holds all preprocessed data.",
        default=cfg.DIR.PREPROCESSED_DATASET,
        type=Path,
    )
    arg_parser.add_argument(
        "--skip",
        dest="skip",
        default=False,
        action="store_true",
        help="If set, previously-processed shapes will be skipped",
    )
    arg_parser.add_argument(
        "--threads",
        dest="num_threads",
        default=mp.cpu_count(),
        help="The number of threads to use to process the data.",
    )
    arg_parser.add_argument(
        "--test",
        "-t",
        dest="test_sampling",
        default=False,
        action="store_true",
        help="If set, the script will produce SDF samples for testing",
    )
    arg_parser.add_argument(
        "--surface",
        dest="surface_sampling",
        default=False,
        action="store_true",
        help="If set, the script will produce mesh surface samples for "
        "evaluation. Otherwise, the script will produce SDF samples "
        "for training.",
    )
    arg_parser.add_argument(
        "--info",
        "-i",
        dest="dataset_info",
        help="Path to file with the taxonomy derived in the work of Choy "
        "et al.",
        default=cfg.DATASET_DESCRIPTION,
    )

    geon_nets.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    geon_nets.configure_logging(args)

    additional_general_args = []
    geon_nets_dir = Path(__file__).absolute().parent.parent
    if args.surface_sampling:
        executable = geon_nets_dir / "bin/SampleVisibleMeshSurface"
        subdir = ws.surface_samples_subdir
        extension = ".ply"
    else:
        executable = geon_nets_dir / "bin/PreprocessMesh"
        subdir = ws.sdf_samples_subdir
        extension = ".npz"

        if args.test_sampling:
            additional_general_args += ["-t"]

    process_meshes(
        args.source_dir,
        args.data_dir,
        args.dataset_info,
        subdir,
        executable,
        args.skip,
        args.surface_sampling,
        extension,
        args.num_threads,
        additional_general_args,
    )


def process_meshes(
    source_dir: Path,
    data_dir: Path,
    dataset_info_path: str,
    target_subdir: str,
    executable: str,
    skip: bool,
    surface_sampling: bool,
    processed_file_extension: str,
    num_threads: int,
    additional_args: t.Sequence[str],
):
    dest_dir = data_dir / target_subdir
    if not dest_dir.exists():
        dest_dir.mkdir(exist_ok=True, parents=True)

    normalization_param_dir: t.Optional[Path] = None
    if surface_sampling:
        normalization_param_dir = data_dir / ws.normalization_param_subdir
        if not normalization_param_dir.exists():
            normalization_param_dir.mkdir()

    mesh_targets_and_specific_args = []
    logging.info(
        "Preprocessing data and placing the results in " + dest_dir.as_posix()
    )

    with open(dataset_info_path) as f:
        dataset_info = json.load(f)

    meshes_classes_to_process = list(dataset_info.keys())

    not_existing_meshes = []
    not_existing_renderings = []
    total_instances = 0

    for class_name in meshes_classes_to_process:
        instances_folders = list((source_dir / class_name).glob("*"))
        logging.debug(
            f"Processing {class_name} with "
            f"{len(instances_folders)} instances"
        )

        total_instances += len(instances_folders)

        for instance_folder in tqdm.tqdm(instances_folders):
            output_shape_dir = dest_dir / class_name / instance_folder.name
            output_shape_dir.mkdir(exist_ok=True, parents=True)
            processed_file_path = (
                output_shape_dir / f"samples{processed_file_extension}"
            )
            if skip and processed_file_path.is_file():
                logging.debug(f"skipping {processed_file_path.as_posix()}")
                continue

            try:
                mesh_filename = (
                    instance_folder / "models" / "model_normalized.obj"
                )
                if not mesh_filename.exists():
                    not_existing_meshes.append(instance_folder.parent.name)
                    continue

                if not (instance_folder / "rendering").exists():
                    not_existing_renderings.append(instance_folder.parent.name)
                    continue

                specific_args = []

                if surface_sampling and normalization_param_dir is not None:
                    normalization_param_target_dir = (
                        normalization_param_dir
                        / class_name
                        / instance_folder.name
                    )
                    if not normalization_param_target_dir.exists():
                        normalization_param_target_dir.mkdir(
                            exist_ok=True, parents=True
                        )
                    normalization_param_file_name = (
                        normalization_param_target_dir
                        / "normalization_params.npz"
                    )
                    specific_args = [
                        "-n",
                        normalization_param_file_name.as_posix(),
                    ]

                mesh_targets_and_specific_args.append(
                    (
                        mesh_filename,
                        processed_file_path.as_posix(),
                        specific_args,
                    )
                )

            except geon_nets.data.NoMeshFileError:
                logging.warning(
                    "No mesh found for instance "
                    + (instance_folder.parent / instance_folder.name)
                )
            except geon_nets.data.MultipleMeshFileError:
                logging.warning(
                    "Multiple meshes found for instance "
                    + (instance_folder.parent / instance_folder.name)
                )
    logging.info(
        f"No shapes ratio: "
        f"{len(not_existing_meshes) / total_instances:.4f}"
    )
    logging.info(
        f"Non normalized and all shapes counts: "
        f"{len(not_existing_meshes)}, {total_instances}"
    )
    logging.info(
        f"Names of classes without shape: " f"{list(set(not_existing_meshes))}"
    )
    logging.info(
        f"Counts of classes without shape: " f"{Counter(not_existing_meshes)}"
    )
    logging.info(
        f"Names of classes without renderings: "
        f"{list(set(not_existing_renderings))}"
    )
    logging.info(
        f"Counts of classes without renderings: "
        f"{Counter(not_existing_renderings)}"
    )
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=int(num_threads)
    ) as executor:

        for mesh_filepath, target_filepath, specific_args in tqdm.tqdm(
            mesh_targets_and_specific_args
        ):
            executor.submit(
                process_mesh,
                mesh_filepath,
                target_filepath,
                executable,
                specific_args + additional_args,
            )

        executor.shutdown()


if __name__ == "__main__":
    main()
