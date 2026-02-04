import keras
import tensorflow as tf

@keras.utils.register_keras_serializable()
class TransposeLayer(keras.layers.Layer):
    def __init__(self, perm=None, **kwargs):
        super().__init__(**kwargs)
        self.perm = perm
    
    def call(self, inputs):
        return tf.transpose(inputs, perm=self.perm)
    
    def get_config(self):
        config = super().get_config()
        config.update({"perm": self.perm})
        return config