import keras
import tensorflow as tf

@keras.utils.register_keras_serializable()
class TransposeLayer(keras.layers.Layer):
    def __init__(self, axes=None, **kwargs):
        super().__init__(**kwargs)
        self.axes = axes
    
    def call(self, inputs):
        return tf.transpose(inputs, perm=self.axes)
    
    def get_config(self):
        config = super().get_config()
        config.update({"axes": self.axes})
        return config