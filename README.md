# ttbar-mva-trainer
A standalone analysis framework for training and evaluating machine learning models for reconstruction of dileptonic ttbar events. The framework includes **pure Python data preprocessing**, data loading and model training using TensorFlow, and integration with the Condor job scheduler for distributed training and evaluation.

The models are designed to perform both reconstruction of the neutrino momenta as well as assignment of jets to the corresponding b-quarks from the top quark decays.

To inject the trained machine learning models into the TopCPToolKit, the models can be exported to ONNX format. Currently, only models providing jet-to-quark assignments as output can be exported to ONNX format for use in the TopCPToolKit.


## Setup

The code can be run in a virtual environment. To set up the virtual environment, you can run the following commands:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Overview
This repository contains a standalone analysis framework for training and evaluating machine learning models on ttbar events.

## Feature Overview

- **Preprocessing**: The preprocessing pipeline is defined in `core/RootPreprocessor.py` and can output to either ROOT or NPZ format for fast I/O. See [Preprocessing](#preprocessing) for details.
- **Data Loading**: The data loading and preprocessing is done using the `DataPreprocessor` class, which is defined in the `core/DataLoader.py` file. This class is used to load the data from preprocessed files and arrange it in a format that can be used for training and evaluation.
- **Reconstruction Models**: The reconstruction models are defined in the `core/reconstruction` directory. The base class for all reconstruction models is the `BaseReconstructor` class, which is defined in the `core/reconstruction/Reconstruction.py` file. Machine learning-based reconstruction models are to be implemented by inheriting from the `KerasFFRecoBase` class, while baseline reconstruction models are to be implemented by inheriting from the `BaselineReconstructor` class.
- **Model Training**: The training of the models is handled by the `KerasFFRecoBase` class which provides methods for training and evaluating machine learning-based reconstruction models.
- **Evaluation**: The evaluation of the models is done using the `ReconstructionPlotter` class, which is defined in the `core/reconstruction/Evaluation.py` file. This class provides methods for evaluating the performance of various reconstruction models using different metrics and visualizations.
- **Export Models for use in TopCPToolKit**: The trained machine learning models can be exported for use in the TopCPToolKit. The `export_to_onnx` method in the `KerasMLWrapper` class is used to export the trained model to a format that can be used in the TopCPToolKit.
- **Condor Integration**: The framework includes integration with the Condor job scheduler for distributed training and evaluation. The Condor scripts are located in the `CONDOR` directory.

## Preprocessing
The preprocessing is now implemented in **pure Python**  and is defined in `core/RootPreprocessor.py`. The Python implementation provides all functionality of the previous C++ preprocessor plus additional features like NPZ output format for faster I/O.

### Quick Start

### Features

The preprocessing pipeline performs:
1. Event pre-selection (lepton/jet multiplicities, charge requirements, truth matching)
2. Particle ordering (leptons by charge, jets by pT)
3. Derived feature computation (invariant masses, ΔR, etc.)
4. Truth information extraction (top/anti-top, neutrinos, ttbar system)
5. Optional NuFlow neutrino reconstruction results
6. Optional initial parton information

### Python API

```python
from core.RootPreprocessor import preprocess_root_file

# Simple preprocessing
data = preprocess_root_file(
    input_path="input.root",
    output_path="output.npz",
    output_format="npz"
)

# Direct integration with DataPreprocessor
from core import DataPreprocessor, LoadConfig

preprocessor = DataPreprocessor(load_config)
preprocessor.load_from_npz("preprocessed_data.npz")
```

For detailed preprocessing documentation, see [`scripts/README_PREPROCESSING.md`](scripts/README_PREPROCESSING.md).


## Data Loading
The `DataPreprocessor` class (in `core/DataLoader.py`) handles loading preprocessed data for training and evaluation. It requires a `LoadConfig` that specifies which features to load from NPZ files and how to interpret them. The DataLoader automatically detects whether data is in flat format (from RootPreprocessor) or structured format and handles the conversion transparently.

### Quick Start

```python
from core.DataLoader import DataPreprocessor
from core.Configs import LoadConfig

# Create LoadConfig specifying which features to load
load_config = LoadConfig(
    jet_features=['pt', 'eta', 'phi', 'e', 'b'],
    lepton_features=['pt', 'eta', 'phi', 'e'],
    met_features=[],
    non_training_features=['truth_ttbar_mass', 'truth_top_mass'],
    jet_truth_label='event_jet_truth_idx',
    lepton_truth_label='event_lepton_truth_idx',
    max_jets=10,
    NUM_LEPTONS=2,
    event_weight='event_weight',
    mc_event_number='mc_event_number',
    neutrino_momentum_features=['px', 'py', 'pz'],
    antineutrino_momentum_features=['px', 'py', 'pz'],
)

# Load data using the config
data_loader = DataPreprocessor(load_config)
data_loader.load_from_npz("preprocessed_data.npz")

# Access data configuration
data_config = data_loader.get_data_config()
print(f"Loaded {data_loader.data_length} events")

# Access features
jet_features = data_loader.feature_data['jet']  # (n_events, max_jets, n_jet_features)
lepton_features = data_loader.feature_data['lepton']  # (n_events, 2, n_lepton_features)
labels = data_loader.feature_data['assignment']  # (n_events, max_jets, 2)

# Split data for training
X_train, y_train, X_test, y_test = data_loader.split_data(test_size=0.2)
```

### Format Auto-Detection

The DataLoader automatically handles two input formats:

1. **Flat format** (from RootPreprocessor): Keys like `lep_pt`, `jet_eta`
   - Automatically groups by particle type based on LoadConfig feature names
   - Builds truth labels from configured truth label keys
   - Constructs regression targets from configured momentum features

2. **Structured format** (legacy): Keys like `lepton`, `jet` with pre-stacked arrays
   - Maintains backward compatibility with existing NPZ files
   - Direct loading without conversion

The LoadConfig tells the DataLoader:
- **Which features to load**: `jet_features`, `lepton_features`, `met_features`
- **Where to find truth labels**: `jet_truth_label`, `lepton_truth_label`
- **Which regression targets to include**: `neutrino_momentum_features`
- **Optional features**: `non_training_features`, `event_weight`, `mc_event_number`

### Complete Integration Example

```python
# Step 1: Preprocess ROOT files
from core.RootPreprocessor import RootPreprocessor

preprocessor = RootPreprocessor()
preprocessor.process_root_file("data.root", "reco")
preprocessor.save_to_npz("data.npz")

# Step 2: Create LoadConfig for your analysis
from core.Configs import LoadConfig

load_config = LoadConfig(
    jet_features=['pt', 'eta', 'phi', 'e', 'b'],
    lepton_features=['pt', 'eta', 'phi', 'e'],
    jet_truth_label='event_jet_truth_idx',
    lepton_truth_label='event_lepton_truth_idx',
    max_jets=10,
    NUM_LEPTONS=2,
    event_weight='event_weight',
    neutrino_momentum_features=['px', 'py', 'pz'],
    antineutrino_momentum_features=['px', 'py', 'pz'],
)

# Step 3: Load for ML training
from core.DataLoader import DataPreprocessor

data_loader = DataPreprocessor(load_config)
data_loader.load_from_npz("data.npz")
X_train, y_train, X_test, y_test = data_loader.split_data()

# Step 4: Train your model
# model.fit(X_train, y_train, validation_data=(X_test, y_test))
```

See [`notebooks/IntegrationExample.ipynb`](notebooks/IntegrationExample.ipynb) for a complete tutorial.

### Advanced Usage

The DataPreprocessor provides additional functionality:
- **Data splitting**: Train/test splits, k-fold cross-validation, even/odd event splitting
- **Feature access**: Get specific features by name or all features by type
- **Custom features**: Add derived features computed from existing data
- **Event weights**: Access and normalize event weights
- **Normalization**: Apply standardization to features

### Key Features

- **Flexible configuration**: LoadConfig specifies exactly what to load
- **Format detection**: Automatically handles flat and structured formats
- **Missing feature handling**: Warnings for missing optional features, errors for required ones
- **Backward compatible**: Works with existing structured NPZ files
- **Type safety**: Validates feature names and array shapes

## Reconstruction Models
The base class for all reconstruction models is the `BaseReconstructor` class, which is defined in the `core/reconstruction/Reconstruction.py` file. This class provides the basic functionality for all reconstruction models regardless of the underlying model type.

### Machine Learning Models
Machine learning-based reconstruction models are to be implemented by inheriting from the `KerasFFRecoBase` class, which is defined in the `core/reconstruction/Reconstruction.py` file. This class provides additional functionality for machine learning-based reconstruction models, such as training and evaluation methods and handling the input and output data formats.

To implement a new machine learning-based reconstruction model, you need to create a new class that inherits from the `KerasFFRecoBase` class and implement the following method:
- `build_model(self, **kwargs)`: This method is used to define the architecture of the machine learning model. You can use TensorFlow/Keras to define the model architecture.

### Baseline Models
Baseline reconstruction models are to be implemented by inheriting from the `BaselineReconstructor` class, which is defined in the `core/assignment_models/BaseLineAssingmentMethods.py` file. This class provides the functionality for baseline reconstruction models, such as simple heuristic-based assignment methods.


## Model Training
The training of the models is handled by the `KerasFFRecoBase` class which provides methods for training and evaluating machine learning-based reconstruction models. The training process involves loading the data using the `DataPreprocessor` class, preprocessing the data, and then training the model using the specified architecture and hyperparameters.


## Evaluation
To evaluate the performance of the reconstruction models, the `ReconstructionPlotter` class is used, which is defined in the `core/reconstruction/Evaluation.py` file. This class provides methods for evaluating the performance of various reconstruction models using different metrics and visualizations. The evaluation process involves loading the test data, making predictions using the trained models, and then calculating various performance metrics such as accuracy, precision, recall, and F1-score. The class also provides methods for visualizing the results using plots and histograms.

To evaluate metrics for machine learning-based reconstructors, the `MLEvaluator` method is used. This method provides functionality for evaluating machine learning-based reconstruction models using various metrics and visualizations.


## Export Models for use in TopCPToolKit
The trained machine learning models can be exported for use in the TopCPToolKit. The `export_to_onnx` method in the `KerasMLWrapper` class is used to export the trained model to a format that can be used in the TopCPToolKit. The exported model can then be integrated into the TopCPToolKit for use in physics analyses. For further use, please refer to the TopCPToolKit documentation and the DiLepTagger module.
Note: So far, only models providing assignments as output can be exported to ONNX format for use in the TopCPToolKit.

## Condor Integration
### Hyperparameter Grid Search
The framework includes integration with the Condor job scheduler for distributed training and evaluation. The Condor scripts are located in the `CONDOR` directory. The `HypParamGridSearch` subdirectory contains scripts for performing hyperparameter grid search using Condor. The `run_training.sh` script is used to run the training of the models using Condor. The `train_hyperparameter.py` script is used to train the models with specified hyperparameters.




## Dependencies
The code is written in Python 3.9 and the dependencies are managed using `pip`. The required dependencies are listed in the `requirements.txt` file. To install the dependencies, you can run the following command:

```bash
pip install -r requirements.txt
```

or if you want to install the dependencies in a virtual environment, you can run the following commands:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
