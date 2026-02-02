import sys

sys.path.append("..")
from core.DataLoader import (
    DataPreprocessor,
    get_load_config_from_yaml,
)
from importlib import reload
import core
import keras as keras
import core.keras_models as keras_models

ROOT_DIR = "/afs/desy.de/user/a/aulich/mva-trainer/"
PLOTS_DIR = ROOT_DIR + "plots/regression_transformer_pil/"
MODEL_DIR = ROOT_DIR + "models/regression_transformer_pil/"
CONFIG_PATH = ROOT_DIR + "CONDOR/training/load_config.yaml"

import os

if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


load_config = get_load_config_from_yaml(CONFIG_PATH)

DataProcessor = DataPreprocessor(load_config)

data_config = DataProcessor.load_from_npz(
    load_config.data_path["nominal"], max_events=10_000_000, event_numbers="even"
)

X_train, y_train = DataProcessor.get_data()

del DataProcessor  # Free memory

Transformer = keras_models.FullRecoTransformer(data_config)


Transformer.build_model(
    hidden_dim=256,
    num_heads=8,
    num_layers=10,
    dropout_rate=0.1,
    compute_HLF=True,
    log_variables=True,
)

Transformer.add_reco_mass_deviation_loss()

Transformer.adapt_normalization_layers(X_train)

Transformer.compile_model(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-4),
    loss={
        "normalized_regression": core.utils.RegressionLoss(),
        "assignment": core.utils.AssignmentLoss(),
    },
    metrics={
        "normalized_regression": core.utils.RegressionDeviation(),
        "assignment": core.utils.AssignmentAccuracy(),
    },
    loss_weights={
        "normalized_regression": 1.0,
        "assignment": 1.0,
        "reco_mass_deviation": 0.1,
    },
)

Transformer.train_model(
    X_train,
    y_train,
    batch_size=1024,
    epochs=100,
    copy_data=False,
    callbacks=[keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
        mode="min",
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        mode="min",
    )],
    validation_split=0.1,
)

Transformer.save_model(MODEL_DIR + "model.keras")