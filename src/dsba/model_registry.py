import joblib
import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
from sklearn.base import BaseEstimator


@dataclass
class ClassifierMetadata:
    id: str
    created_at: str
    algorithm: str
    hyperparameters: dict[str, Any]
    target_column: str
    description: str
    performance_metrics: dict[str, float]


def save_model(model: BaseEstimator, metadata: ClassifierMetadata) -> None:
    model_path = _get_model_path(metadata.id)
    model_metadata_path = _get_model_metadata_path(metadata.id)
    logging.info("Save model to path: " + str(model_path))
    joblib.dump(model, model_path)
    with open(model_metadata_path, "w") as f:
        json.dump(asdict(metadata), f)


def list_models_ids() -> list[str]:
    models_dir = _get_models_dir()
    model_files = _list_pickle_files(models_dir)
    models_ids = [_remove_file_extension(model) for model in model_files]
    return models_ids


#creation of function which lists the datasets
# DATASET_DIR = "data"
# def list_datasets():
#         datasets = [f for f in os.listdir(DATASET_DIR) if f.endswith(".csv")]
#         return datasets


def load_model(model_id) -> BaseEstimator:
    model_path = _get_model_path(model_id)
    return _load_model_from_path(model_path)


def load_model_metadata(model_id) -> ClassifierMetadata:
    metadata_path = _get_model_metadata_path(model_id)
    with open(metadata_path, "r") as f:
        metadata_as_dict = json.load(f)
    metadata = ClassifierMetadata(**metadata_as_dict)
    return metadata


def _load_model_from_path(path: str | Path) -> BaseEstimator:
    # A method starting with underscore is by convention a private method
    # meaning it is not intended to be used outside of this file
    # It is optional but facilitates the readability of the code
    path = _get_absolute_path(path)
    logging.info("Load model from path: " + str(path))
    return joblib.load(path)


def _get_model_metadata_path(model_id: str) -> Path:
    models_dir = _get_models_dir()
    metadata_path = os.path.join(models_dir, f"{model_id}.json")
    return Path(metadata_path)


def _get_model_path(model_id: str) -> Path:
    models_dir = _get_models_dir()
    model_path = os.path.join(models_dir, f"{model_id}.pkl")
    return Path(model_path)


def _get_models_dir() -> Path:
    # We can't use a specific path here, it will be different on each machine.
    # We could ask the client code where the models are every time they want to use this module.
    # It would be fine. Instead, we use what is called an "environment variable".
    # It is a way for the operating system to provide parameters to a program.
    # Make sure to set this environment variable before calling this code.
    DSBA_MODELS_ROOT_PATH = os.getenv("DSBA_MODELS_ROOT_PATH")
    if DSBA_MODELS_ROOT_PATH is None:
        raise ValueError(
            "Environment variable DSBA_MODELS_ROOT_PATH is not set. "
            "Please set it to the root path of the models."
        )
    models_dir = _get_absolute_path(DSBA_MODELS_ROOT_PATH)
    if not Path(models_dir).exists():
        logging.info(
            f"Parent directory for models does not exist, creating it at {models_dir}"
        )
        Path(models_dir).mkdir(parents=True)
    return models_dir


def _list_pickle_files(path: Path) -> list[str]:
    """List all files in a directory that end with .pkl"""
    return [f for f in os.listdir(path) if f.endswith(".pkl")]


def _get_absolute_path(path: str | Path) -> Path:
    """
    This is a purely technical function, it does not change the meaning of the path.
    But for example, one of the issues it prevents:
    if you just use "~/my/directory/and/file" some naive code may actually create a file with this name, starting with the caracter "~"
    and not understand that this is supposed to refer to the user's home directory
    """
    return Path(path).expanduser().resolve(strict=False)


def _remove_file_extension(file_name: str) -> str:
    return os.path.splitext(file_name)[0]
