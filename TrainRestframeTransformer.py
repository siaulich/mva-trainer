import sys
import argparse
import os
import numpy as np
import keras as keras
import matplotlib.pyplot as plt
import yaml
import tensorflow as tf
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from copy import deepcopy

from core import keras_models
from core import utils


@dataclass
class TrainConfig:
    batch_size: int = 1024
    epochs: int = 50
    callbacks: Optional[Dict[str, any]] = field(default_factory=dict)
    validation_split: float = 0.1
    shuffle: bool = True
    verbose: int = 1


@dataclass
class ModelConfig:
    model_type: str = "FeatureConcatTransformer"
    model_options: Dict[str, any] = field(default_factory=dict)
    model_params: Dict[str, any] = field(default_factory=dict)
    compile_options: Dict[str, any] = field(default_factory=dict)


from core.DataLoader import (
    DataPreprocessor,
    DataConfig,
    LoadConfig,
)


def parse_args():
    """Parse command line arguments for running the training script."""
    parser = argparse.ArgumentParser(
        description="Train Transformer model with specified hyperparameters"
    )

    # Configuration file arguments
    parser.add_argument(
        "--load_config",
        type=str,
        required=True,
        help="Path to the load configuration YAML file",
    )
    parser.add_argument(
        "--train_config",
        type=str,
        required=True,
        help="Path to the training configuration YAML file",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        required=True,
        help="Path to the model configuration YAML file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save trained models and logs",
    )

    parser.add_argument(
        "--event_numbers",
        type=str,
        default="all",
        help="Comma-separated list of event numbers to use for training (optional)",
    )

    parser.add_argument(
        "--max_events",
        type=int,
        default=None,
        help="Maximum number of events to load from the dataset (optional)",
    )

    return parser.parse_args()


def load_yaml_config(file_path):
    """Load a YAML configuration file."""
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":
    args = parse_args()

    load_config = LoadConfig(**load_yaml_config(args.load_config)["LoadConfig"])
    train_config = TrainConfig(**load_yaml_config(args.train_config)["TrainConfig"])
    model_config = ModelConfig(**load_yaml_config(args.model_config)["ModelConfig"])

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    data_preprocessor = DataPreprocessor(load_config)
    data_config = data_preprocessor.load_from_npz(
        load_config.data_path,
        event_numbers=args.event_numbers,
        max_events=args.max_events,
    )
    X, y = data_preprocessor.get_data()

    model = keras_models._get_model(model_config.model_type)
    model = model(data_config)

    build_options = model_config.model_options
    build_options.update(**model_config.model_params)

    model.build_model(**(build_options))

    model.adapt_normalization_layers(X)

    model.compile_model(
        optimizer=keras.optimizers.get(model_config.compile_options["optimizer"]),
        loss={
            "assignment": utils.AssignmentLoss(),
            "normalized_regression": utils.RestframeLoss(),
        },
        metrics={
            "assignment": utils.AssignmentAccuracy(),
        },
        loss_weights=model_config.compile_options.get("loss_weights", None),
        add_physics_informed_loss=model_config.compile_options.get(
            "add_physics_informed_loss", False
        ),
    )

    train_options = deepcopy(train_config.__dict__)

    callbacks = []
    for callback_name, callback_params in train_config.callbacks.items():
        callback_class = getattr(keras.callbacks, callback_name)
        callbacks.append(callback_class(**callback_params))
    train_options["callbacks"] = callbacks

    y["regression"] = np.concatenate(
        [
            utils.lorentz_vector_from_neutrino_momenta_array(y["regression"]),
            utils.lorentz_vector_from_neutrino_momenta_array(y["regression"])
            + X["lepton_truth"],
        ],
        axis=1,
    )

    X_train, y_train, sample_weights = model.prepare_training_data(
        X, y, copy_data=True, sample_weights=utils.compute_sample_weights(X, y)
    )

    even_history = model.train_model(
        X_train,
        y_train,
        **train_options,
    )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.event_numbers == "even":
        model.save_model(os.path.join(args.output_dir, f"odd_model.keras"))
        model.export_to_onnx(os.path.join(args.output_dir, "odd_model.onnx"))
    elif args.event_numbers == "odd":
        model.save_model(os.path.join(args.output_dir, f"even_model.keras"))
        model.export_to_onnx(os.path.join(args.output_dir, "even_model.onnx"))
    else:
        model.save_model(os.path.join(args.output_dir, f"model.keras"))
        model.export_to_onnx(os.path.join(args.output_dir, "model.onnx"))

    with open(os.path.join(args.output_dir, "model_config.yaml"), "w") as file:
        yaml.dump(
            {
                "LoadConfig": load_config.__dict__,
                "TrainConfig": train_config.__dict__,
                "ModelConfig": model_config.__dict__,
            },
            file,
        )
