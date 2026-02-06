import sys
import argparse
import os
import numpy as np
import keras as keras
import matplotlib.pyplot as plt
import yaml
import tensorflow as tf
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from copy import deepcopy

from core import keras_models
from core import utils
from core import DataConfig, LoadConfig


def load_yaml_config(file_path):
    """Load a YAML configuration file."""
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


@dataclass
class RecontructorConfig:
    type: str = "KerasFFRecoBase"
    options: Dict[str, any] = field(default_factory=dict)

@dataclass
class EvaluationConfig:
    reconstructors: List[RecontructorConfig] = field(default_factory=list)
    evaluation_event_numbers: str = "even"
    evaluation_sample_size: Optional[int] = None
    output_dir: str = "evaluation_results"