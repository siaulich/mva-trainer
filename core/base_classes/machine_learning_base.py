from abc import ABC, abstractmethod
from . import BaseUtilityModel
from core.DataLoader import DataConfig
import keras as keras
import numpy as np
import tensorflow as tf
import tf2onnx
import onnx
import os
from core.components import (
    onnx_support,
    GenerateMask,
    InputPtEtaPhiELayer,
    InputMetPhiLayer,
    ComputeHighLevelFeatures,
    PhysicsInformedLoss,
)
import core.components as components
import core.utils as utils
from copy import deepcopy


@keras.utils.register_keras_serializable()
class KerasModelWrapper(keras.Model):
    def predict_dict(self, x, batch_size=2048, verbose=0, steps=None, **kwargs):
        predictions = super().predict(
            x, batch_size=batch_size, verbose=verbose, steps=steps, **kwargs
        )
        if not isinstance(predictions, dict):
            if not isinstance(predictions, list):
                predictions = [predictions]
            return dict(zip(self.output_names, predictions))
        return predictions


class KerasMLWrapper(BaseUtilityModel, ABC):
    def __init__(
        self,
        config: DataConfig,
        name="ml_assigner",
        assignment_name=None,
        full_reco_name=None,
        model_id=None,
    ):
        """
        Initializes the AssignmentBaseModel class.
        Args:
            data_preprocessor (DataPreprocessor): An instance of the DataPreprocessor class
                that provides preprocessed data and metadata required for model initialization.
        """
        self.model: KerasModelWrapper = None
        self.trainable_model: KerasModelWrapper = None
        self.history = None
        self.sample_weights = None
        self.class_weights = None
        self.NUM_LEPTONS = config.NUM_LEPTONS
        self.max_jets = config.max_jets
        self.met_features = config.met_features
        self.n_jets: int = len(config.jet_features)
        self.n_leptons: int = len(config.lepton_features)
        self.n_met: int = len(config.met_features) if config.met_features else 0
        self.n_global: int = (
            len(config.global_event_features) if config.has_global_event_features else 0
        )
        self.padding_value: float = config.padding_value
        self.feature_index_dict = config.feature_indices
        self.perform_regression = False

        # initialize empty dicts to hold inputs and transformed inputs
        self.inputs = {}
        self.transformed_inputs = {}
        self.normed_inputs = {}
        self.masks = {}
        self.model_id = None

        # Use assignment_name and full_reco_name if provided, otherwise fall back to name
        if assignment_name is None:
            assignment_name = name
        if full_reco_name is None:
            full_reco_name = name

        super().__init__(
            config=config,
            assignment_name=assignment_name,
            full_reco_name=full_reco_name,
        )

    def build_model(self, **kwargs):
        raise NotImplementedError("Subclasses must implement build_model method.")
        pass

    def prepare_training_data(
        self, data, sample_weights=None, class_weights=None, copy_data=False
    ):
        self.sample_weights = sample_weights
        self.class_weights = class_weights
        for input_name in self.inputs.keys():
            if input_name not in data:
                raise ValueError(f"Input '{input_name}' not found in data dictionary.")

        if copy_data:
            X_train = {key: deepcopy(data[key]) for key in self.inputs.keys()}
        else:
            X_train = {key: data[key] for key in self.inputs.keys()}
        y_train = {}

        # Rename targets to match model output names
        y_train["assignment"] = data["assignment"]
        y_train["regression"] = data["regression"]
        if not self.perform_regression:
            y_train.pop("regression")

        if "regression" in y_train:
            regression_data = y_train.pop("regression")
            upscale_layer = self.model.get_layer("regression")
            if upscale_layer is None:
                raise ValueError(
                    "Regression layer not found in model. Cannot prepare regression targets."
                )
            if not isinstance(upscale_layer, keras.layers.Rescaling):
                raise ValueError(
                    "Regression layer is not a Rescaling layer. Cannot prepare regression targets."
                )
            regression_std = upscale_layer.scale
            regression_mean = upscale_layer.offset
            y_train["normalized_regression"] = (
                regression_data - regression_mean
            ) / regression_std

        if "reco_mass_deviation" in self.trainable_model.output_names:
            y_train["reco_mass_deviation"] = np.zeros(
                (y_train["assignment"].shape[0], 1), dtype=np.float32
            )

        return X_train, y_train, sample_weights

    def add_reco_mass_deviation_loss(self):
        if self.model is None:
            raise ValueError(
                "Model has not been built yet. Call build_model() before adding physics-informed loss. If you use regression, adapt the normalization layers first."
            )
        if "regression" not in self.model.output_names:
            raise ValueError(
                "Regression output not found in model outputs. Cannot add physics-informed loss."
            )
        reco_mass_deviation_layer = PhysicsInformedLoss(name="reco_mass_deviation")
        neutrino_momenta = self.model.get_layer("regression").output
        assignment_probs = self.model.get_layer("assignment").output
        normalised_neutrino_output = self.trainable_model.get_layer(
            "normalized_regression"
        ).output

        if neutrino_momenta is None:
            raise ValueError(
                "Regression output not found in model outputs. Cannot add physics-informed loss."
            )

        lepton_momenta = self.inputs["lep_inputs"]

        reco_mass_deviation = reco_mass_deviation_layer(
            neutrino_momenta, lepton_momenta
        )
        self.trainable_model = keras.Model(
            inputs=self.model.inputs,
            outputs={
                "assignment": assignment_probs,
                "normalized_regression": normalised_neutrino_output,
                "reco_mass_deviation": reco_mass_deviation,
            },
        )

        print("Added physics-informed loss to the model.")

    def _prepare_inputs(
        self, log_variables=True, compute_HLF=False, use_global_event_features=False, transform_inputs=True
    ):
        jet_inputs = keras.Input(shape=(self.max_jets, self.n_jets), name="jet_inputs")
        lep_inputs = keras.Input(
            shape=(self.NUM_LEPTONS, self.n_leptons), name="lep_inputs"
        )
        met_inputs = keras.Input(shape=(1, self.n_met), name="met_inputs")

        if self.config.has_global_event_features:
            global_event_inputs = keras.Input(
                shape=(self.n_global,),
                name="global_event_inputs",
            )

        # Generate jet mask
        jet_mask = GenerateMask(padding_value=-999, name="jet_mask")(jet_inputs)

        # Transform inputs
        transformed_jet_inputs = InputPtEtaPhiELayer(
            name="jet_input_transform",
            log_variables=log_variables,
            padding_value=self.padding_value,
        )(jet_inputs)
        transformed_lep_inputs = InputPtEtaPhiELayer(
            name="lep_input_transform",
            log_variables=log_variables,
            padding_value=self.padding_value,
        )(lep_inputs)
        transformed_met_inputs = InputMetPhiLayer(name="met_input_transform")(
            met_inputs
        )

        if compute_HLF:
            high_level_features = ComputeHighLevelFeatures(
                name="compute_high_level_features",
                padding_value=self.padding_value,
            )(
                jet_input=transformed_jet_inputs,
                lepton_input=transformed_lep_inputs,
                jet_mask=jet_mask,
            )

        # Normalize inputs
        if transform_inputs:
            normed_jet_inputs = keras.layers.Normalization(name="jet_input_normalization")(
                transformed_jet_inputs
            )
            normed_lep_inputs = keras.layers.Normalization(name="lep_input_normalization")(
                transformed_lep_inputs
            )
            normed_met_inputs = keras.layers.Normalization(name="met_input_normalization")(
                transformed_met_inputs
            )
        else:
            normed_jet_inputs = keras.layers.Normalization(name="jet_input_normalization")(
                jet_inputs
            )
            normed_lep_inputs = keras.layers.Normalization(name="lep_input_normalization")(
                lep_inputs
            )
            normed_met_inputs = keras.layers.Normalization(name="met_input_normalization")(
                met_inputs
            )

        if self.config.has_global_event_features and use_global_event_features:
            print("Adding normalization for global event features")
            normed_global_event_inputs = keras.layers.Normalization(
                name="global_event_input_normalization"
            )(global_event_inputs)
        self.inputs = {
            "jet_inputs": jet_inputs,
            "lep_inputs": lep_inputs,
            "met_inputs": met_inputs,
        }
        self.transformed_inputs = {
            "jet_inputs": transformed_jet_inputs,
            "lepton_inputs": transformed_lep_inputs,
            "met_inputs": transformed_met_inputs,
        }
        self.normed_inputs = {
            "jet_inputs": normed_jet_inputs,
            "lepton_inputs": normed_lep_inputs,
            "met_inputs": normed_met_inputs,
        }
        self.masks = {"jet_mask": jet_mask}

        if compute_HLF:
            normed_hlf_inputs = keras.layers.Normalization(
                name="hlf_input_normalization", axis=(-1, -2)
            )(high_level_features)
            self.transformed_inputs["hlf_inputs"] = high_level_features
            self.normed_inputs["hlf_inputs"] = normed_hlf_inputs

        if self.config.has_global_event_features and use_global_event_features:
            self.inputs["global_event_inputs"] = global_event_inputs
            self.transformed_inputs["global_event_inputs"] = global_event_inputs
            self.normed_inputs["global_event_inputs"] = normed_global_event_inputs

        return self.normed_inputs, self.masks

    def train_model(
        self,
        data,
        epochs,
        batch_size,
        sample_weights=None,
        validation_split=0.2,
        callbacks=[],
        copy_data=False,
        **kwargs,
    ):
        if self.model is None:
            raise ValueError(
                "Model has not been built yet. Call build_model() before train_model()."
            )

        X_train, y_train, sample_weights = self.prepare_training_data(
            data, sample_weights=sample_weights, copy_data=copy_data
        )
        if self.trainable_model is None:
            self.trainable_model = self.model

        callbacks = callbacks if callbacks is not None else []
        callbacks.append(
            keras.callbacks.TerminateOnNaN()
        )  # Ensure training stops on NaN loss

        if self.history is None:
            self.history = self.trainable_model.fit(
                X_train,
                y_train,
                sample_weight=sample_weights,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                **kwargs,
            )
        else:
            append_history = self.trainable_model.fit(
                X_train,
                y_train,
                sample_weight=sample_weights,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                **kwargs,
            )
            # Append new history to existing history
            for key in append_history.history:
                self.history.history[key].extend(append_history.history[key])
            self.history.epoch.extend(
                [e + self.history.epoch[-1] + 1 for e in append_history.epoch]
            )
        return self.history

    def save_model(self, file_path="model.keras"):
        """
        Saves the current model to the specified file path in Keras format and writes the model's structure to a text file.
        Args:
            file_path (str): The file path where the model will be saved.
                            Must end with ".keras". Defaults to "model.keras".
        """

        if self.model is None:
            raise ValueError(
                "Model not built. Please build the model using build_model() method."
            )
        if ".keras" not in file_path:
            raise ValueError(
                "File path must end with .keras. Please provide a valid file path."
            )
        self.model.save(file_path)

        if self.history is not None:
            history_path = file_path.replace(".keras", "_history")
            np.savez(history_path, **self.history.history)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path, model_id=None):
        """
        Loads a pre-trained Keras model from the specified file path.

        This method uses custom objects defined in the `CustomObjects` class to
        ensure that any custom layers, loss functions, or metrics used in the
        model are properly loaded.

        Args:
            file_path (str): The file path to the saved Keras model.
        """

        custom_objects = {
            name: obj
            for name, obj in zip(components.__dict__.items(), utils.__dict__.items())
            if isinstance(obj, type) and issubclass(obj, keras.layers.Layer)
        }
        custom_objects.update({"KerasModelWrapper": KerasModelWrapper})
        self.model_id = model_id

        self.model = keras.saving.load_model(file_path, custom_objects=custom_objects)
        print(f"Model loaded from {file_path}")
        history_path = file_path.replace(".keras", "_history.npz")
        if os.path.exists(history_path):
            loaded_history = np.load(history_path, allow_pickle=True)
            history_dict = {key: loaded_history[key].tolist() for key in loaded_history}
            self.history = keras.callbacks.History()
            self.history.history = history_dict
            print(f"Training history loaded from {history_path}")
        else:
            print(f"WARNING: No training history found at {history_path}")

    def adapt_normalization_layers(self, data: dict):
        """
        Adapts the normalization layers in a functional model.
        Each normalization layer is adapted using data that has been passed
        through all preceding layers (up to but excluding the normalization layer).
        """
        # --- Prepare and unpad jet data ---
        jet_data = data["jet_inputs"]  # (num_events, n_jets, n_features)
        jet_mask = np.any(jet_data != self.padding_value, axis=-1)
        unpadded_jet_data = jet_data[jet_mask]
        num_jets = unpadded_jet_data.shape[0]
        num_events = num_jets // self.max_jets
        unpadded_jet_data = unpadded_jet_data[: num_events * self.max_jets, :].reshape(
            (num_events, self.max_jets, self.n_jets)
        )
        lep_data = data["lep_inputs"][:num_events, :, :]
        met_data = data["met_inputs"][:num_events, :, :]
        if self.config.has_global_event_features:
            global_event_data = data["global_event_inputs"][:num_events, :]

        # --- Helper: build a submodel up to (but not including) a target layer ---
        def get_pre_norm_submodel(model, target_layer_name):
            target_layer = model.get_layer(target_layer_name)
            # find which input(s) feed into this layer
            inbound_nodes = target_layer._inbound_nodes
            if not inbound_nodes:
                raise ValueError(f"Layer '{target_layer_name}' has no inbound nodes.")
            inbound_tensors = inbound_nodes[0].input_tensors
            submodel = keras.Model(
                inputs=model.inputs,
                outputs=(
                    inbound_tensors if len(inbound_tensors) > 1 else inbound_tensors[0]
                ),
                name=f"pre_{target_layer_name}_model",
            )
            return submodel

        input_data = {
            "jet_inputs": unpadded_jet_data,
            "lep_inputs": lep_data,
            "met_inputs": met_data,
        }

        if self.config.has_global_event_features:
            input_data["global_event_inputs"] = global_event_data

        # --- Loop over normalization layers and adapt each ---
        for layer in self.model.layers:
            if isinstance(layer, keras.layers.Normalization):
                submodel = get_pre_norm_submodel(self.model, layer.name)
                submodel_inputs = submodel.inputs
                if not isinstance(submodel_inputs, list):
                    submodel_inputs = [submodel_inputs]

                submodel_input_data = {}
                for input_tensor in submodel_inputs:
                    input_name = input_tensor.name.split(":")[0]
                    if input_name not in input_data:
                        raise ValueError(
                            f"Input '{input_name}' required for submodel not found in input_data."
                        )
                    submodel_input_data[input_name] = input_data[input_name]
                # Get transformed data from submodel
                transformed_data = submodel.predict(
                    submodel_input_data,
                    batch_size=1024,
                    verbose=0,
                )
                layer.adapt(transformed_data)
                del submodel
                print("Adapted normalization layer: ", layer.name)

        if (
            self.perform_regression
            and "normalized_regression" in self.model.output_names
        ):
            # neutrino_truth_std = np.std(data["regression"], axis=0)
            # neutrino_truth_mean = np.mean(data["regression"], axis=0)
            denormalisation_layer = keras.layers.Rescaling(
                scale=1e5, # Scale neutrinos to 100 GeV by default
                # offset=neutrino_truth_mean,
                # scale = neutrino_truth_std,
                name="regression",
            )
            self.model = KerasModelWrapper(
                inputs=self.model.inputs,
                outputs={
                    "assignment": self.model.get_layer("assignment").output,
                    "regression": denormalisation_layer(
                        self.model.get_layer("normalized_regression").output
                    ),
                },
            )
            print("Set regression denormalization layer with computed mean and std.")

    def export_to_onnx(self, onnx_file_path="model.onnx"):
        """
        Exports the current Keras model to onnx format.

        The model is wrapped to take a flattened input tensor and split it into the original inputs.

        Saves a .txt file with model input names and positions for reference.

        Args:
            onnx_file_path (str): The file path where the ONNX model will be saved.
                                  Must end with ".onnx". Defaults to "model.onnx".
        Raises:
            ValueError: If the model has not been built (i.e., `self.model` is None).
            ValueError: If the provided file path does not end with ".onnx".
        """
        if self.model is None:
            raise ValueError(
                "Model not built. Please build the model using build_model() method."
            )
        if ".onnx" not in onnx_file_path:
            raise ValueError(
                "File path must end with .onnx. Please provide a valid file path."
            )

        # Define input shapes
        jet_shape = (self.max_jets, self.n_jets)
        lep_shape = (self.NUM_LEPTONS, self.n_leptons)
        input_shapes = [jet_shape, lep_shape]
        if self.n_met > 0:
            met_shape = (1, self.n_met)
            input_shapes.append(met_shape)
        if (
            self.config.has_global_event_features
            and "global_event_inputs" in self.model.input
        ):
            global_event_shape = (self.n_global,)
            input_shapes.append(global_event_shape)

        # Create a new model that takes a flat input and splits it
        flat_input_size = sum(np.prod(shape) for shape in input_shapes)
        flat_input = keras.Input(shape=(flat_input_size,), name="flat_input")

        split_layer = onnx_support.SplitInputsLayer(input_shapes)
        split_inputs = split_layer(flat_input)

        model_outputs = self.model([*split_inputs])

        wrapped_model = keras.Model(inputs=flat_input, outputs=model_outputs)

        # Convert to ONNX
        spec = (tf.TensorSpec((None, flat_input_size), tf.float32, name="flat_input"),)
        onnx_model,_ = tf2onnx.convert.from_keras(
            wrapped_model,
            input_signature=spec,   
        )

        meta = onnx_model.metadata_props.add()
        meta.key = "NN_padding_value"
        meta.value = str(self.padding_value)

        meta = onnx_model.metadata_props.add()
        meta.key = "NN_max_jets"
        meta.value = str(self.max_jets)

        # Save ONNX model
        onnx.save_model(onnx_model, onnx_file_path)


        print(f"ONNX model saved to {onnx_file_path}")
