# ttbar-2l-mva-trainer
A standalone analysis framework for training and evaluating machine learning models for reconstruction of dileptonic ttbar events. The framework includes data-preprocessing, data loading and model training using TensorFlow, and integration with the Condor job scheduler for distributed training and evaluation.

The models are designed to perform both reconstruction of the neutrino momenta as well as assignment of jets to the corresponding b-quarks from the top quark decays.

To inject the trained machine learning models into the TopCPToolKit, the models can be exported to ONNX format. Currently, only models feed-forward neural network architectures are supported for export to ONNX format, but support for additional architectures can be implemented.


## Setup

The code can be run in a virtual environment. To set up the virtual environment, you can run the following commands:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Preprocessing
The preprocessing is now implemented in  and is defined in `core/RootPreprocessor.py`. 
The `RootPreprocessor` class handles the preprocessing of the data and can output to either ROOT or NPZ format for fast I/O. The preprocessing pipeline includes steps such as feature engineering and handling of missing values. The preprocessed data can then be used for training and evaluation of the reconstruction models.


## Data Loading
The `DataPreprocessor` class (in `core/DataLoader.py`) handles loading preprocessed data for training and evaluation. It requires a `LoadConfig` that specifies which features to load from NPZ files and how to interpret them. The DataLoader automatically detects whether data is in flat format (from RootPreprocessor) or structured format and handles the conversion transparently.


## Machine Learning Models
Machine learning-based reconstruction models can be implemented by inheriting from the `KerasMLWrapper` class, which is defined in the `core/base_classes/keras_ml_wrapper.py` file. This class provides the basic funcationality for machine learning-based reconstruction models.


### Model Architectures
Depending on the type of architecture used for the machine learning-based reconstruction model, the `KerasMLWrapper` class can be further extended.
For a feed-forward neural network architecture, the `KerasFFRecoBase` class can be used, which is defined in the `core/reconstruction/keras_ff_reco_base.py` file. This class provides additional functionality specific to feed-forward neural network architectures, such as methods for building the model architecture and training the model. To implement a feed-forward neural network-based reconstruction model, you can create a new class that inherits from the `KerasFFRecoBase` class and implement `build_model` method to define the architecture of the model.

### Model Training
The training of the machine learning-based reconstruction models is handled by the `KerasMLWrapper` class, which provides methods for training and evaluating the models.


## Evaluation
### Baseline Models
Baseline reconstruction models are to be implemented by inheriting from the `BaselineReconstructor` class, which is defined in the `core/assignment_models/BaseLineAssingmentMethods.py` file. This class provides the functionality for baseline reconstruction models, such as simple heuristic-based assignment methods.

### Plotting and Metrics
To evaluate the performance of the reconstruction models, the `ReconstructionPlotter` class is used, which is defined in the `core/reconstruction/reconstruction_evaluator.py` file. This class provides methods for evaluating the performance of various reconstruction models using different metrics and visualizations. The evaluation process involves loading the test data, making predictions using the trained models, and then calculating various performance metrics such as accuracy, precision, recall, and F1-score. The class also provides methods for visualizing the results using plots and histograms.

### Machine Learning-Based Reconstruction Evaluation
To evaluate metrics for machine learning-based reconstructors, the `MLEvaluator` method is used. This method provides functionality for evaluating machine learning-based reconstruction models using various metrics and visualizations.


## Condor Integration
### Hyperparameter Grid Search
The framework includes integration with the Condor job scheduler for distributed training and evaluation. The Condor scripts are located in the `CONDOR` directory.


## Dependencies
The code is written in Python 3.9 and the dependencies are managed using `pip`. The required dependencies are listed in the `requirements.txt` file. To install the dependencies, you can run the following command:
