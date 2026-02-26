from .assignment_RNN import *
from .assignment_transformer import *
from .regression_transformer import *
from ..reconstruction.baseline_methods import *

def _get_model(model_name) -> type[KerasFFRecoBase]:
    if model_name not in globals():
        raise ValueError(f"Model '{model_name}' not found in keras_models.")
    if not issubclass(globals()[model_name], KerasFFRecoBase):
        raise TypeError(f"Model '{model_name}' is not a subclass of KerasFFRecoBase.")
    return globals()[model_name]