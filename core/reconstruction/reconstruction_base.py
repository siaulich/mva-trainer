import tensorflow as tf
import keras as keras
import numpy as np
from abc import ABC, abstractmethod
from core.DataLoader import DataConfig
from core.base_classes import BaseUtilityModel, KerasMLWrapper, KerasModelWrapper
from core.components import (
    OutputUpScaleLayer,
    PhysicsInformedLoss,
    ConfidenceLossOutputLayer,
)
from core.utils import losses


class EventReconstructorBase(BaseUtilityModel, ABC):
    def __init__(
        self,
        config: DataConfig,
        assignment_name,
        full_reco_name,
        neutrino_name=None,
        perform_regression=True,
        use_nu_flows=True,
    ):
        BaseUtilityModel.__init__(
            self,
            config=config,
            assignment_name=assignment_name,
            full_reco_name=full_reco_name,
            neutrino_name=neutrino_name,
        )
        self.max_jets = config.max_jets
        self.NUM_LEPTONS = config.NUM_LEPTONS
        if perform_regression and not config.has_neutrino_truth:
            print(
                "WARNING: perform_regression is set to True, but config.has_neutrino_truth is False. Setting perform_regression to False."
            )
            perform_regression = False
        if use_nu_flows and not config.has_nu_flows_neutrino_truth:
            print(
                "WARNING: use_nu_flows is set to True, but config.use_nu_flows is False. Setting use_nu_flows to False."
            )
            use_nu_flows = False
        if perform_regression and use_nu_flows:
            print(
                "WARNING: perform_regression is set to True, but use_nu_flows, is also True. Setting use_nu_flows False to make us of neutrino regression implementation."
            )

        self.perform_regression = perform_regression
        self.use_nu_flows = use_nu_flows

    def predict_indices(self, data_dict):
        pass

    def reconstruct_neutrinos(self, data_dict: dict[str : np.ndarray]):
        if self.perform_regression:
            raise NotImplementedError(
                "This method should be implemented in subclasses that perform regression."
            )
        if self.use_nu_flows:
            if "nu_flows_neutrino_truth" in data_dict:
                return data_dict["nu_flows_neutrino_truth"]
            print(
                "WARNING: use_nu_flows is True but 'nu_flows_neutrino_truth' not found in data_dict. Falling back to 'neutrino_truth'."
            )
        if "regression" in data_dict:
            return data_dict["regression"]
        print(f"data_dict keys: {list(data_dict.keys())}")
        raise ValueError(
            "No regression targets found in data_dict for neutrino reconstruction."
        )

    def evaluate_accuracy(self, data_dict, true_labels, per_event=False):
        """
        Evaluates the model's performance on the provided data and true indices.

        Args:
            data_dict (dict): A dictionary containing input data for the model.
            true_labels (np.ndarray): The true labels (one-hot) to compare against the model's predictions.
        Returns:
            float | np.ndarray: The accuracy of the model's predictions. If per_event is True,
            returns an array of booleans indicating correctness per event; otherwise, returns overall accuracy.
        """
        predictions = self.predict_indices(data_dict)
        return self.compute_accuracy(predictions, true_labels, per_event)

    def evaluate_regression(self, data_dict, true_values):
        """
        Evaluates the regression performance of the model on the provided data and true values.

        Args:
            data_dict (dict): A dictionary containing input data for the model.
            true_values (np.ndarray): The true regression target values to compare against the model's predictions.
        Returns:
            float: The mean squared error of the model's regression predictions.
        """
        predicted_values = self.reconstruct_neutrinos(data_dict)
        return self.compute_regression_mse(predicted_values, true_values)

    def compute_accuracy(self, pred_values, true_values, per_event=False):
        predicted_indices = np.argmax(pred_values, axis=-2)
        true_indices = np.argmax(true_values, axis=-2)
        if per_event:
            correct_predictions = np.all(predicted_indices == true_indices, axis=-1)
            return correct_predictions
        else:
            correct_predictions = np.all(predicted_indices == true_indices, axis=-1)
            accuracy = np.mean(correct_predictions)
            return accuracy

    def compute_regression_mse(self, pred_values, true_values):
        relative_errors = (pred_values - true_values) / np.where(
            true_values != 0, true_values, 1
        )
        mse = np.mean(np.square(relative_errors))
        return mse

    def evaluate(self, data_dict):
        results = {}
        if "assignment_truth" in data_dict:
            accuracy = self.evaluate_accuracy(
                data_dict, data_dict["assignment_truth"], per_event=False
            )
            results["accuracy"] = accuracy
        if self.perform_regression and "regression" in data_dict:
            mse = self.evaluate_regression(data_dict, data_dict["regression"])
            results["regression_mse"] = mse
        return results


class KerasFFRecoBase(EventReconstructorBase, KerasMLWrapper):
    def __init__(
        self,
        config: DataConfig,
        name,
        perform_regression=False,
        use_nu_flows=True,
        load_model_path=None,
    ):
        EventReconstructorBase.__init__(
            self,
            config=config,
            assignment_name=name,
            full_reco_name=(
                name
                if perform_regression
                else name + (r" + $\nu^2$-Flows" if use_nu_flows else r" + True $\nu$")
            ),
            perform_regression=perform_regression,
            use_nu_flows=use_nu_flows,
        )
        KerasMLWrapper.__init__(
            self, config=config, perform_regression=perform_regression
        )
        self.model: keras.models.Model = None
        self.trainable_model: keras.models.Model = None
        if load_model_path is not None:
            self.load_model(load_model_path)
        self.predict_confidence = False

    def _build_model_base(
        self,
        jet_assignment_probs,
        regression_output=None,
        confidence_score=None,
        **kwargs,
    ):
        trainable_outputs = {}
        outputs = {}
        outputs["assignment"] = jet_assignment_probs
        trainable_outputs["assignment"] = jet_assignment_probs

        if self.perform_regression and regression_output is None:
            raise ValueError(
                "perform_regression is True but no regression_output provided to build_model."
            )
        if self.perform_regression and regression_output is not None:
            outputs["normalized_regression"] = regression_output
            trainable_outputs["normalized_regression"] = regression_output
        if confidence_score is not None:
            outputs["confidence_score"] = confidence_score
            trainable_outputs["confidence_loss_output"] = ConfidenceLossOutputLayer(
                name="confidence_loss_output"
            )(jet_assignment_probs, confidence_score)
            self.predict_confidence = True
        self.model = keras.models.Model(
            inputs=self.inputs,
            outputs=outputs,
            name=kwargs.get("name", "reco_model"),
        )
        self.trainable_model = keras.models.Model(
            inputs=self.inputs,
            outputs=trainable_outputs,
            name=kwargs.get("name", "reco_trainable_model"),
        )

    def compile_model(
        self, loss, optimizer, metrics=None, add_physics_informed_loss=False, **kwargs
    ):
        if self.trainable_model is None:
            raise ValueError(
                "Model has not been built yet. Call build_model() before compile_model()."
            )
        if self.predict_confidence:
            loss["confidence_loss_output"] = losses.ConfidenceScoreLoss()
        if add_physics_informed_loss:
            self.add_reco_mass_deviation_loss()
            print(
                "Compiling model with physics informed loss. Ensure that the loss dictionary includes 'reco_mass_deviation'."
            )
            if "reco_mass_deviation" not in loss:
                loss["reco_mass_deviation"] = lambda y_true, y_pred: y_pred
        self.trainable_model.compile(
            loss=loss, optimizer=optimizer, metrics=metrics, **kwargs
        )

    def generate_one_hot_encoding(self, predictions, exclusive):
        """
        Generates a one-hot encoded array from the model's predictions.
        This method processes the raw predictions from the model and converts them
        into a one-hot encoded format, indicating the associations between jets and leptons.
        Args:
            predictions (np.ndarray): The raw predictions from the model, typically
                of shape (batch_size, max_jets, NUM_LEPTONS).
            exclusive (bool): If True, ensures exclusive assignments between jets
                and leptons
        Returns:
            np.ndarray: A one-hot encoded array of shape (batch_size, max_jets, 2),
            where the last dimension represents the association between jets and
            leptons. The value 1 indicates an association, and 0 indicates no association.
        """
        prediction_product_matrix = predictions[..., 0][:,:,np.newaxis] + predictions[..., 1][:, np.newaxis, ...] # shape (batch_size, max_jets, max_jets)
        if exclusive:
            prediction_product_matrix[:,np.arange(predictions.shape[1]),np.arange(predictions.shape[1])] = 0 # set diagonal to zero to enforce exclusivity
        one_hot = np.zeros((predictions.shape[0], self.max_jets, self.NUM_LEPTONS), dtype=int)
        idx = np.argmax(prediction_product_matrix.reshape(predictions.shape[0], -1), axis=1)
        one_hot[np.arange(predictions.shape[0]), np.unravel_index(idx, prediction_product_matrix.shape[1:])[0], 0] = 1
        one_hot[np.arange(predictions.shape[0]), np.unravel_index(idx, prediction_product_matrix.shape[1:])[1], 1] = 1

        if False: # old implementation, kept for reference
            one_hot = np.zeros((predictions.shape[0], self.max_jets, 2), dtype=int)
            for i in range(predictions.shape[0]):
                probs = predictions[i].copy()
                for _ in range(self.NUM_LEPTONS):
                    jet_index, lepton_index = np.unravel_index(
                        np.argmax(probs), probs.shape
                    )
                    one_hot[i, jet_index, lepton_index] = 1
                    probs[jet_index, :] = 0
                    probs[:, lepton_index] = 0
        return one_hot

    def predict_indices(self, data: dict[str : np.ndarray], exclusive=True):
        """
        Predicts the indices of jets and leptons based on the model's predictions.
        This method processes the predictions from the model and returns a one-hot
        encoded array indicating the associations between jets and leptons.
        Args:
            data (dict): A dictionary containing input data for prediction. It should
                include keys "jet_inputs" and "lep_inputs", and optionally "met_inputs" if met
                features are used by the model.
            exclusive (bool, optional): If True, ensures exclusive assignments between
                jets and leptons, where each jet is assigned to at most one lepton and
                vice versa. Defaults to True.
        Returns:
            np.ndarray: A one-hot encoded array of shape (batch_size, max_jets, 2),
            where the last dimension represents the association between jets and
            leptons. The value 1 indicates an association, and 0 indicates no association.
        Raises:
            ValueError: If the model is not built (i.e., `self.model` is None).
        """

        if self.model is None:
            raise ValueError(
                "Model not built. Please build the model using build_model() method."
            )

        return self.complete_forward_pass(data)[0]

    def reconstruct_neutrinos(self, data: dict[str : np.ndarray]):
        """
        Reconstructs neutrino kinematics based on the model's regression output.
        This method processes the regression output from the model and returns
        the reconstructed neutrino kinematics.
        Args:
            data (dict): A dictionary containing input data for prediction. It should
                include keys "jet_inputs" and "lep_inputs", and optionally "met_inputs" if met
                features are used by the model.
        Returns:
            np.ndarray: An array containing the reconstructed neutrino kinematics.
        Raises:
            ValueError: If the model is not built (i.e., `self.model` is None) or
                if regression targets are not specified in the config.
        """

        if self.model is None:
            raise ValueError(
                "Model not built. Please build the model using build_model() method."
            )

        return self.complete_forward_pass(data)[1]

    def predict(self, data: dict[str : np.ndarray], batch_size=2048, verbose=0):
        inputs = {}
        for key in self.model.input:
            if hasattr(key, "name"):
                input_name = key.name.split(":")[0]
            elif isinstance(key, str):
                input_name = key
            else:
                raise ValueError(
                    f"Unexpected input key type: {type(key)}. Expected a Keras tensor or string."
                )
            if input_name in data:
                inputs[input_name] = data[input_name]
            else:
                raise ValueError(
                    f"Expected input '{input_name}' not found in data dictionary."
                )
        predictions = self.model.predict(inputs, batch_size=batch_size, verbose=verbose)
        return predictions

    def complete_forward_pass(self, data: dict[str : np.ndarray]):
        """
        Performs a complete forward pass through the model, returning both
        jet-lepton assignment predictions and neutrino kinematics reconstruction.
        This method processes the input data through the model and returns
        both the assignment predictions and the reconstructed neutrino kinematics.
        Args:
            data (dict): A dictionary containing input data for prediction. It should
                include keys "jet_inputs" and "lep_inputs", and optionally "met_inputs" if met
                features are used by the model.
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - A one-hot encoded array of shape (batch_size, max_jets, 2),
                  representing jet-lepton assignments.
                - An array containing the reconstructed neutrino kinematics.
        """

        if self.model is None:
            raise ValueError(
                "Model not built. Please build the model using build_model() method."
            )
        predictions = self.predict(data)
        assignment_predictions = self.generate_one_hot_encoding(
            predictions["assignment"], exclusive=True
        )
        if "regression" in predictions:
            neutrino_reconstruction = predictions["regression"]
        else:
            neutrino_reconstruction = EventReconstructorBase.reconstruct_neutrinos(
                self, data
            )
        return assignment_predictions, neutrino_reconstruction

    def evaluate(self, data_dict):
        assignment_predictions, regression_predictions = self.complete_forward_pass(
            data_dict
        )
        results = {}
        if "assignment" in data_dict:
            accuracy = self.compute_accuracy(
                assignment_predictions, data_dict["assignment"], per_event=False
            )
            results["accuracy"] = accuracy
        if self.perform_regression and "regression" in data_dict:
            mse = self.compute_regression_mse(
                regression_predictions, data_dict["regression"]
            )
            results["regression_mse"] = mse
        return results
