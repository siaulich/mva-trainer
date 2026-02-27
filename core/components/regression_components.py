import keras as keras
import tensorflow as tf


@keras.utils.register_keras_serializable()
class SplitTransformerOutput(keras.layers.Layer):
    def __init__(
        self, name="SplitTransformerOutput", max_jets=6, max_leptons=2, **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.max_jets = max_jets
        self.max_leptons = max_leptons

    def call(self, inputs):
        """
        Splits the transformer output into jet, lepton, and MET components.
        Args:
            inputs (tf.Tensor): The output tensor from the transformer of shape (batch_size, seq_len, hidden_dim).
        Returns:
            tuple: A tuple containing:
                - jet_outputs (tf.Tensor): The jet component of shape (batch_size, max_jets, hidden_dim).
                - lepton_outputs (tf.Tensor): The lepton component of shape (batch_size, max_leptons, hidden_dim).
                - met_outputs (tf.Tensor): The MET component of shape (batch_size, 1, hidden_dim).
        """
        jet_outputs = inputs[:, : self.max_jets, :]
        lepton_outputs = inputs[:, self.max_jets : self.max_jets + self.max_leptons, :]
        met_outputs = inputs[:, -1:, :]
        return jet_outputs, lepton_outputs, met_outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_jets": self.max_jets,
                "max_leptons": self.max_leptons,
            }
        )
        return config


@keras.utils.register_keras_serializable()
class ConcatLeptonCharge(keras.layers.Layer):
    def __init__(self, name="ConcatLeptonCharge", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, input):
        """
        Concatenates lepton charge information to lepton features.
        Args:
            input: tensor of shape (batch_size, max_leptons, feature_dim) where the last dimension is the charge.

        Returns:
            tf.Tensor: Tensor of shape (batch_size, max_leptons, feature_dim + 1) with charge concatenated.
        """
        lepton_features = input[:, :, :-1]
        # Create charge indicators: 1 for first lepton, -1 for second lepton
        charge_indicators = tf.constant([[1.0], [-1.0]], dtype=lepton_features.dtype)
        charge_indicators = tf.reshape(charge_indicators, [1, 2, 1])
        charge_indicators = tf.tile(charge_indicators, [tf.shape(input)[0], 1, 1])
        concatenated = tf.concat([lepton_features, charge_indicators], axis=-1)
        return concatenated


@keras.utils.register_keras_serializable()
class ExpandJetMask(keras.layers.Layer):
    def __init__(self, name="ExpandJetMask", extra_sequence_length=0, **kwargs):
        super().__init__(name=name, **kwargs)
        self.extra_sequence_length = extra_sequence_length

    def call(self, jet_mask):
        """
        Expands the jet mask to match the sequence length of the transformer output.
        Args:
            jet_mask (tf.Tensor): The jet mask tensor of shape (batch_size, max_jets).
        Returns:
            tf.Tensor: The expanded jet mask tensor of shape (batch_size, max_jets + extra_sequence_length).
        """
        batch_size = tf.shape(jet_mask)[0]
        extra_mask = tf.ones(
            (batch_size, self.extra_sequence_length), dtype=jet_mask.dtype
        )
        expanded_mask = tf.concat([jet_mask, extra_mask], axis=1)
        return expanded_mask

    def get_config(self):
        config = super().get_config()
        config.update({"extra_sequence_length": self.extra_sequence_length})
        return config


@keras.utils.register_keras_serializable()
class UnbinRegressionOutput(keras.layers.Layer):
    def __init__(self, scale, n_bins=None, name="UnbinRegressionOutput", **kwargs):
        super().__init__(name=name, **kwargs)
        self.scale = scale
        self.n_bins = n_bins

    def build(self, input_shape):
        self.n_bins = input_shape[-1]
        super().build(input_shape)

    def call(self, inputs):
        """
        Unbins the regression output by applying the inverse of the scaling factor.
        Args:
            inputs (tf.Tensor): The binned regression output tensor of shape (batch_size, ..., n_bins).
        Returns:
            tf.Tensor: The unbinned regression output tensor of shape (batch_size, ...).
        """
        dtype = self.compute_dtype
        scale = tf.cast(self.scale, dtype)
        selected_bins = tf.cast(tf.argmax(inputs, axis=-1), inputs.dtype) / tf.cast(self.n_bins, inputs.dtype)
        unbinned_output = (selected_bins - 0.5) * scale
        return unbinned_output

    def get_config(self):
        config = super().get_config()
        config.update({"scale": keras.saving.serialize_keras_object(self.scale)})
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config = config.copy()
        config["scale"] = keras.saving.deserialize_keras_object(
            config["scale"], custom_objects=custom_objects
        )
        return cls(**config)

    def bin_data(self, regression_data):
        """
        Bins the regression data by applying the scaling factor and converting to bin indices.
        Args:
            regression_data (tf.Tensor): The continuous regression data tensor of shape (batch_size, ...).
        Returns:
            tf.Tensor: The binned regression data tensor of shape (batch_size, ..., n_bins).
        """
        scaled_data = (regression_data / self.scale + 0.5) * self.n_bins
        binned_data = tf.cast(
            tf.clip_by_value(scaled_data, 0, self.n_bins - 1), tf.int32
        )
        one_hot_binned_data = tf.one_hot(binned_data, depth=self.n_bins)
        return one_hot_binned_data
