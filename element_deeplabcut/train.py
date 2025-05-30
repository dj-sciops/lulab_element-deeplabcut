"""
Module for training DeepLabCut models within DataJoint.
----------------------------
This module leverages DeepLabCut's built-in training functionality and captures the
results in DataJoint tables.
The user is expected to make changes to the DeepLabCut-generated config.yaml file and
the pytorch-config.yaml file to specify the training parameters prior to running this module.
"""

import datajoint as dj
import inspect
import importlib
import re
from pathlib import Path
import yaml

from element_interface.utils import find_full_path, dict_to_uuid
from .readers import dlc_reader

schema = dj.schema()
_linking_module = None


def activate(
    train_schema_name: str,
    *,
    create_schema: bool = True,
    create_tables: bool = True,
    linking_module: str = None,
):
    """Activate this schema.

    Args:
        train_schema_name (str): schema name on the database server
        create_schema (bool): when True (default), create schema in the database if it
                            does not yet exist.
        create_tables (bool): when True (default), create schema tables in the database
                             if they do not yet exist.
        linking_module (str): a module (or name) containing the required dependencies.

    Dependencies:
    Functions:
        get_dlc_root_data_dir(): Returns absolute path for root data director(y/ies)
                                 with all behavioral recordings, as (list of) string(s).
        get_dlc_processed_data_dir(): Optional. Returns absolute path for processed
                                      data. Defaults to session video subfolder.
    """

    if isinstance(linking_module, str):
        linking_module = importlib.import_module(linking_module)
    assert inspect.ismodule(
        linking_module
    ), "The argument 'dependency' must be a module's name or a module"
    assert hasattr(
        linking_module, "get_dlc_root_data_dir"
    ), "The linking module must specify a lookup function for a root data directory"

    global _linking_module
    _linking_module = linking_module

    # activate
    schema.activate(
        train_schema_name,
        create_schema=create_schema,
        create_tables=create_tables,
        add_objects=_linking_module.__dict__,
    )


# -------------- Functions required by element-deeplabcut ---------------


def get_dlc_root_data_dir() -> list:
    """Pulls relevant func from parent namespace to specify root data dir(s).

    It is recommended that all paths in DataJoint Elements stored as relative
    paths, with respect to some user-configured "root" director(y/ies). The
    root(s) may vary between data modalities and user machines. Returns a full path
    string or list of strings for possible root data directories.
    """
    root_directories = _linking_module.get_dlc_root_data_dir()
    if isinstance(root_directories, (str, Path)):
        root_directories = [root_directories]

    if (
        hasattr(_linking_module, "get_dlc_processed_data_dir")
        and get_dlc_processed_data_dir() not in root_directories
    ):
        root_directories.append(_linking_module.get_dlc_processed_data_dir())

    return root_directories


def get_dlc_processed_data_dir() -> str:
    """Pulls relevant func from parent namespace. Defaults to DLC's project /videos/.

    Method in parent namespace should provide a string to a directory where DLC output
    files will be stored. If unspecified, output files will be stored in the
    session directory 'videos' folder, per DeepLabCut default.
    """
    if hasattr(_linking_module, "get_dlc_processed_data_dir"):
        return _linking_module.get_dlc_processed_data_dir()
    else:
        return get_dlc_root_data_dir()[0]


# ----------------------------- Table declarations ----------------------


@schema
class DLCTrainingTask(dj.Manual):
    """Table for creating a DLC training task.

    Attributes:
        task_id (int): Primary key. Unique identifier for the training task.
        project_path (str): Path to the DeepLabCut project directory.
        dlc_config (longblob): DeepLabCut-generated config.yaml file.
        pytorch_config (longblob): DeepLabCut-generated pytorch-config.yaml file.
        shuffle (int): Shuffle for the training task.
        trainingsetindex (int): Index of the training fraction in config.yaml.
        snapshot_file (filepath): Optional. Path to latest snapshot file if available.
    """

    definition = """
    task_id: int
    ---
    project_path: varchar(255)  # Path to the DeepLabCut project directory
    dlc_config: longblob
    pytorch_config: longblob
    shuffle: int
    trainingsetindex: int
    snapshot_file=null: filepath@dlc-training
    """


@schema
class DLCModelTraining(dj.Computed):
    """Table for training DeepLabCut models.

    Attributes:
        task_id (int): Foreign key to task_id in TrainingTask table.
        trained_config (longblob): DeepLabCut-generated config.yaml file after training.
        trained_pytorch_config (longblob): DeepLabCut-generated pytorch-config.yaml file
        after training.
        training_log_file (filepath): Path to the train.txt file.
        training_snapshot_file (filepath): Path to the latest snapshot file after training.
    """

    definition = """
    -> DLCTrainingTask
    ---
    trained_pose_cfg: longblob
    trained_pytorch_config: longblob
    training_log_file: filepath@dlc-training
    training_snapshot_file: filepath@dlc-training
    """

    def make(self, key):
        """Run model training after verifying that the config files match what was
        ingested in the TrainingTask table."""
        from deeplabcut.compat import train_network
        import yaml
        import pathlib

        # Fetch the task entry from TrainingTask
        project_dir, dlc_config_db, pytorch_config_db = (DLCTrainingTask & key).fetch1(
            "project_path", "dlc_config", "pytorch_config"
        )
        
        # Locate the model folder config files
        dlc_config_path = get_dlc_root_data_dir() / (project_dir + "config.yaml")
        pytorch_config_path = get_dlc_root_data_dir() / (
            project_dir + "pytorch-config.yaml"
        )

        # Load the model folder config files
        with open(dlc_config_path, "r") as f:
            dlc_config_file = yaml.safe_load(f)
        with open(pytorch_config_path, "r") as f:
            pytorch_config_file = yaml.safe_load(f)

        # Compare the contents
        if dlc_config_db != dlc_config_file:
            raise ValueError(
                f"Contents of DLC config file: {dlc_config_path} do not match the database config file."
            )
        if pytorch_config_db != pytorch_config_file:
            raise ValueError(
                f"Contents of PyTorch config file: {pytorch_config_path} do not match the database config file."
            )

        # Proceed with training if files match
        trainingsetindex, shuffle = (TrainingTask & key).fetch1(
            "trainingsetindex", "shuffle"
        )

        train_network(
            config=dlc_config_path.as_posix(),
            shuffle=shuffle,
            trainingsetindex=trainingsetindex,
        )

        # Fetch the trained pose config and pytorch config
        iteration = dlc_config_file["iteration"]
        training_dir_path = pathlib.Path(
            project_dir + f"/dlc-pytorch-models/iteration-{iteration}/"
        )
        trained_config_path = next(
            (get_dlc_processed_data_dir() / training_dir_path).rglob(
                "*/train/pose_cfg.yaml"
            )
        )
        trained_pytorch_config_path = next(
            (get_dlc_processed_data_dir() / training_dir_path).rglob(
                "*/train/pytorch-config.yaml"
            )
        )
        training_log_filepath = next(
            (get_dlc_processed_data_dir() / training_dir_path).glob("*/train/train.txt")
        )
        training_snapshot_file = sorted(
            (get_dlc_processed_data_dir() / training_dir_path).glob(
                "*/train/snapshot_*.pth"
            )
        )[-1]

        # Insert the results into ModelTraining table
        self.insert1(
            {
                **key,
                "trained_pose_cfg": yaml.safe_load(trained_config_path),
                "trained_pytorch_config": yaml.safe_load(trained_pytorch_config_path),
                "training_log_file": training_log_filepath.relative_to(
                    get_dlc_processed_data_dir()
                ),
                "training_snapshot_file": training_snapshot_file.relative_to(
                    get_dlc_processed_data_dir()
                ),
            }
        )
