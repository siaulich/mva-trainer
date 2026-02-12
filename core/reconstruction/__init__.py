"""
Refactored evaluation module for ML-based reconstruction.

This module provides clean, modular tools for evaluating event reconstruction
methods with support for:
- Feature importance analysis
- Accuracy metrics with bootstrap confidence intervals
- Binned performance analysis
- Complementarity analysis between methods
- Mass resolution calculations
- Comprehensive visualization tools
"""
from .reconstruction_base import EventReconstructorBase, KerasFFRecoBase
from .BaselineMethods import *
from .ground_truth_reconstructor import GroundTruthReconstructor, PerfectAssignmentReconstructor, CompositeNeutrinoComponentReconstructor


__all__ = [
    # Main evaluators
    "MLEvaluator",
    "ReconstructionPlotter",
    
    # Feature importance
    "FeatureImportanceCalculator",
    
    # Base utilities
    "PlotConfig",
    "BootstrapCalculator",
    "BinningUtility",
    "FeatureExtractor",
    "AccuracyCalculator",
    
    # Plotting utilities
    "AccuracyPlotter",
    "ConfusionMatrixPlotter",
    "ResolutionPlotter",
    
    # Physics calculations
    "TopReconstructor",
    "ResolutionCalculator",
]