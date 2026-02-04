from .Configs import DataConfig, LoadConfig

try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    print("Could not set TensorFlow GPU memory growth.")
    pass

