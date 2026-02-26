"""Plotting utilities for evaluation."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional, List

from .evaluator_utils import (
    PlotConfig,
    BinningUtility,
    BootstrapCalculator,
    FeatureExtractor,
)


class AccuracyPlotter:
    """Handles plotting of accuracy metrics."""

    @staticmethod
    def plot_overall_accuracies(
        reconstructor_names: List[str],
        accuracies: List[Tuple[float, float, float]],
        config: PlotConfig,
    ):
        """
        Plot overall accuracies with error bars.

        Args:
            reconstructor_names: List of reconstructor names
            accuracies: List of (mean, lower, upper) tuples
            config: Plot configuration

        Returns:
            Tuple of (figure, axis)
        """
        names = reconstructor_names
        means = [acc[0] for acc in accuracies]
        lowers = [acc[1] for acc in accuracies]
        uppers = [acc[2] for acc in accuracies]

        errors_lower = [(mean - lower) for mean, lower in zip(means, lowers)]
        errors_upper = [(upper - mean) for mean, upper in zip(means, uppers)]

        fig, ax = plt.subplots(figsize=config.figsize)
        x_pos = np.arange(len(names))

        ax.bar(
            x_pos,
            means,
            yerr=[errors_lower, errors_upper],
            capsize=5,
            alpha=0.7,
            ecolor="black",
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("Assignment Accuracy")
        ax.set_title(
            f"Assignment Accuracy of Jet Reconstructors ({config.confidence*100:.0f}% CI)"
        )
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=config.alpha)
        

        return fig, ax

    @staticmethod
    def plot_binned_accuracy(
        bin_centers: np.ndarray,
        binned_accuracies: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
        reconstructor_names: List[str],
        bin_counts: np.ndarray,
        bins: np.ndarray,
        feature_label: str,
        config: PlotConfig,
        show_combinatoric: bool = True,
        combinatoric_accuracy: Optional[np.ndarray] = None,
    ):
        """Plot binned accuracy vs. a feature."""
        fig, ax = plt.subplots(figsize=config.figsize)
        color_map = plt.get_cmap("tab10")
        fmt_map = ["o", "s", "D", "^", "v", "P", "*", "X", "h", "8"]

        # Plot combinatoric baseline if requested
        if show_combinatoric and combinatoric_accuracy is not None:
            ax.plot(
                bin_centers,
                combinatoric_accuracy,
                label="Random Assignment",
                color="black",
                linestyle="--",
            )

        # Plot each reconstructor
        for index, (name, (mean_acc, lower, upper)) in enumerate(
            zip(reconstructor_names, binned_accuracies)
        ):
            if config.show_errorbar:
                errors_lower = mean_acc - lower
                errors_upper = upper - mean_acc
                ax.errorbar(
                    bin_centers,
                    mean_acc,
                    yerr=[errors_lower, errors_upper],
                    fmt=fmt_map[index % len(fmt_map)],
                    label=name,
                    color=color_map(index),
                    linestyle="None",
                )
            else:
                ax.plot(
                    bin_centers,
                    mean_acc,
                    label=name,
                    color=color_map(index),
                )

        # Configure main axes
        ax.set_xlabel(feature_label)
        ax.set_ylabel("Assignment accuracy")
        ax.set_ylim(0, 1)
        ax.set_xlim(bins[0], bins[-1])
        ax.grid(alpha=config.alpha)
        ax.legend(loc="best")

        # Add event count histogram
        AccuracyPlotter._add_count_histogram(ax, bin_centers, bin_counts, bins)

        # Set title
        title = f"Assignment accuracy per Bin vs {feature_label}"
        if config.show_errorbar:
            title += f" ({config.confidence*100:.0f}% CI)"
        ax.set_title(title)

        
        return fig, ax

    @staticmethod
    def _add_count_histogram(ax, bin_centers, bin_counts, bins):
        """Add event count histogram to plot."""
        ax_twin = ax.twinx()
        ax_twin.bar(
            bin_centers,
            bin_counts,
            width=(bins[1] - bins[0]),
            alpha=0.2,
            color="red",
            # label="Event Count",
        )
        ax_twin.set_ylabel("Normalised Event Count", color="red")
        ax_twin.tick_params(axis="y", labelcolor="red")

    @staticmethod
    def plot_feature_assignment_success(
        bin_centers: np.ndarray,
        correct_hist: np.ndarray,
        incorrect_hist: np.ndarray,
        feature_label: str,
        figsize: Tuple[int, int] = (8, 6),
    ):
        """Plot histogram of correct vs incorrect assignments per feature bin."""
        fig, ax = plt.subplots(figsize=figsize)

        width = bin_centers[1] - bin_centers[0]

        ax.bar(
            bin_centers,
            correct_hist,
            width=width,
            label="Correct Assignments",
            color="green",
            alpha=0.7,
        )
        ax.bar(
            bin_centers,
            incorrect_hist,
            width=width,
            label="Incorrect Assignments",
            color="red",
            alpha=0.7,
        )

        ax.set_xlabel(feature_label)
        ax.set_ylabel("a. u.")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)

        
        return fig, ax


class SelectionAccuracyPlotter:
    """Handles plotting of selection accuracy metrics."""

    @staticmethod
    def plot_selection_accuracies(
        reconstructor_names: List[str],
        accuracies: List[Tuple[float, float, float]],
        config: PlotConfig,
    ):
        """
        Plot selection accuracies with error bars.

        Args:
            reconstructor_names: List of reconstructor names
            accuracies: List of (mean, lower, upper) tuples
            config: Plot configuration

        Returns:
            Tuple of (figure, axis)
        """
        names = reconstructor_names
        means = [acc[0] for acc in accuracies]
        lowers = [acc[1] for acc in accuracies]
        uppers = [acc[2] for acc in accuracies]

        errors_lower = [(mean - lower) for mean, lower in zip(means, lowers)]
        errors_upper = [(upper - mean) for mean, upper in zip(means, uppers)]

        fig, ax = plt.subplots(figsize=config.figsize)
        x_pos = np.arange(len(names))

        ax.bar(
            x_pos,
            means,
            yerr=[errors_lower, errors_upper],
            capsize=5,
            alpha=0.7,
            ecolor="black",
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("Selection Accuracy")
        ax.set_title(
            f"Selection Accuracy ({config.confidence*100:.0f}% CI)"
        )
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=config.alpha)
        

        return fig, ax

    def plot_binned_selection_accuracy(
        bin_centers: np.ndarray,
        binned_accuracies: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
        reconstructor_names: List[str],
        bin_counts: np.ndarray,
        bins: np.ndarray,
        feature_label: str,
        config: PlotConfig,
        show_combinatoric: bool = True,
        combinatoric_accuracy: Optional[np.ndarray] = None,
    ):
        """Plot binned accuracy vs. a feature."""
        fig, ax = plt.subplots(figsize=config.figsize)
        color_map = plt.get_cmap("tab10")
        fmt_map = ["o", "s", "D", "^", "v", "P", "*", "X", "h", "8"]

        # Plot combinatoric baseline if requested
        if show_combinatoric and combinatoric_accuracy is not None:
            ax.plot(
                bin_centers,
                combinatoric_accuracy,
                label="Random Selection",
                color="black",
                linestyle="--",
            )

        # Plot each reconstructor
        for index, (name, (mean_acc, lower, upper)) in enumerate(
            zip(reconstructor_names, binned_accuracies)
        ):
            if config.show_errorbar:
                errors_lower = mean_acc - lower
                errors_upper = upper - mean_acc
                ax.errorbar(
                    bin_centers,
                    mean_acc,
                    yerr=[errors_lower, errors_upper],
                    fmt=fmt_map[index % len(fmt_map)],
                    label=name,
                    color=color_map(index),
                    linestyle="None",
                )
            else:
                ax.plot(
                    bin_centers,
                    mean_acc,
                    label=name,
                    color=color_map(index),
                )

        # Configure main axes
        ax.set_xlabel(feature_label)
        ax.set_ylabel("Jet Selection Accuracy")
        ax.set_ylim(0, 1)
        ax.set_xlim(bins[0], bins[-1])
        ax.grid(alpha=config.alpha)
        ax.legend(loc="best")

        # Add event count histogram
        AccuracyPlotter._add_count_histogram(ax, bin_centers, bin_counts, bins)

        # Set title
        title = f"Selection Accuracy per Bin vs {feature_label}"
        if config.show_errorbar:
            title += f" ({config.confidence*100:.0f}% CI)"
        ax.set_title(title)

        
        return fig, ax


class ConfusionMatrixPlotter:
    """Handles plotting of confusion matrices."""

    @staticmethod
    def plot_confusion_matrices(
        true_labels: np.ndarray,
        predictions_list: List[np.ndarray],
        reconstructor_names: List[str],
        normalize: bool = True,
        figsize_per_plot: Tuple[int, int] = (5, 5),
    ):
        """
        Plot confusion matrices for all reconstructors.

        Args:
            true_labels: True assignment labels
            predictions_list: List of prediction arrays
            reconstructor_names: List of reconstructor names
            normalize: Whether to normalize the confusion matrix
            figsize_per_plot: Size of each subplot

        Returns:
            Tuple of (figure, axes)
        """
        from sklearn.metrics import confusion_matrix

        n_reconstructors = len(reconstructor_names)
        rows = int(np.ceil(np.sqrt(n_reconstructors)))
        cols = int(np.ceil(n_reconstructors / rows))

        fig, axes = plt.subplots(
            cols,
            rows,
            figsize=(figsize_per_plot[0] * rows, figsize_per_plot[1] * cols),
        )
        axes = axes.flatten() if n_reconstructors > 1 else [axes]

        true_indices = np.argmax(true_labels, axis=-2).flatten()

        for i, (name, predictions) in enumerate(
            zip(reconstructor_names, predictions_list)
        ):
            predicted_indices = np.argmax(predictions, axis=-2).flatten()

            cm = confusion_matrix(
                true_indices,
                predicted_indices,
                normalize="true" if normalize else None,
            )

            sns.heatmap(
                cm,
                annot=True,
                fmt=".2f" if normalize else "d",
                ax=axes[i],
                cmap="Blues",
                cbar_kws={"label": "Normalized Count" if normalize else "Count"},
            )
            axes[i].set_title(f"{name}")
            axes[i].set_xlabel("Predicted Label")
            axes[i].set_ylabel("True Label")

        # Remove unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        fig.suptitle("Confusion Matrix")
        
        return fig, axes[: i + 1]

    @staticmethod
    def plot_variable_confusion_matrix(
        true_values: np.ndarray,
        predicted_values: np.ndarray,
        variable_label: str,
        axes: plt.Axes,
        bins: np.ndarray,
        normalize: Optional[str] = None,
        plot_mean=False,
        **kwargs,
    ):
        """Plot confusion matrix for a specific variable."""

        hist, xedges, yedges = np.histogram2d(
            true_values, predicted_values, bins=[bins, bins]
        )

        if normalize == "true":
            hist = hist / (hist.sum(axis=1, keepdims=True) + 1e-6)
        elif normalize == "pred":
            hist = hist / (hist.sum(axis=0, keepdims=True)  + 1e-6)
        elif normalize == "all":
            hist = hist / (hist.sum() + 1e-6)
        else:
            pass  # No normalization

        im = axes.imshow(
            hist.T,
            origin="lower",
            cmap="viridis",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            aspect="auto",
        )

        if plot_mean:
            bin_centers_x = 0.5 * (xedges[:-1] + xedges[1:])
            avg_y = []
            for i in range(len(xedges) - 1):
                mask = (true_values.flatten() >= xedges[i]) & (
                    true_values.flatten() < xedges[i + 1]
                )
                if np.sum(mask) > 0:
                    avg_y.append(np.mean(predicted_values.flatten()[mask]))
                else:
                    avg_y.append(np.nan)
            axes.plot(
            bin_centers_x,
            avg_y,
            color="red",
            marker="o",
            linestyle="--",
            label=f"Mean Prediction",
            )

        axes.set_xlabel(f"{variable_label} Truth")
        axes.set_ylabel(f"{variable_label} Prediction")
        axes.get_figure().colorbar(im, ax=axes)


class ResolutionPlotter:
    """Handles plotting of mass resolution metrics."""

    @staticmethod
    def plot_binned_resolution(
        bin_centers: np.ndarray,
        binned_resolutions: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
        reconstructor_names: List[str],
        bin_counts: np.ndarray,
        bins: np.ndarray,
        feature_label: str,
        resolution_label: str,
        config: PlotConfig,
    ):
        """Plot binned resolution vs. a feature."""
        fig, ax = plt.subplots(figsize=config.figsize)
        color_map = plt.get_cmap("tab10")
        fmt_map = ["o", "s", "D", "^", "v", "P", "*", "X", "h", "8"]
        # Plot each reconstructor
        for index, (name, (mean_res, lower, upper)) in enumerate(
            zip(reconstructor_names, binned_resolutions)
        ):

            if config.show_errorbar:
                errors_lower = mean_res - lower
                errors_upper = upper - mean_res
                ax.errorbar(
                    bin_centers,
                    mean_res,
                    yerr=[errors_lower, errors_upper],
                    fmt=fmt_map[index % len(fmt_map)],
                    label=name,
                    color=color_map(index),
                    linestyle="None",
                )
            else:
                ax.plot(
                    bin_centers,
                    mean_res,
                    label=name,
                    color=color_map(index),
                )

        # Configure axes
        ax.set_xlabel(feature_label)
        ax.set_ylabel(resolution_label)
        ax.set_xlim(bins[0], bins[-1])
        ax.grid(alpha=config.alpha)
        ax.legend(loc="best")

        # Add event count histogram
        ax_twin = ax.twinx()
        ax_twin.bar(
            bin_centers,
            bin_counts,
            width=(bins[1] - bins[0]),
            alpha=0.2,
            color="red",
            label="Event Count",
        )
        ax_twin.set_ylabel("Event Count", color="red")
        ax_twin.tick_params(axis="y", labelcolor="red")

        # Set title
        title = f"{resolution_label} per Bin vs {feature_label}"
        if config.show_errorbar:
            title += f" ({config.confidence*100:.0f}% CI)"
        ax.set_title(title)

        
        return fig, ax


class NeutrinoDeviationPlotter:
    """Handles plotting of neutrino regression deviation metrics."""

    @staticmethod
    def plot_overall_deviations(
        reconstructor_names: List[str],
        deviations: List[Tuple[float, float, float]],
        config: PlotConfig,
    ):
        """
        Plot overall neutrino deviations with error bars.

        Args:
            reconstructor_names: List of reconstructor names
            deviations: List of (mean, lower, upper) tuples
            config: Plot configuration

        Returns:
            Tuple of (figure, axis)
        """
        names = reconstructor_names
        means = [dev[0] for dev in deviations]
        lowers = [dev[1] for dev in deviations]
        uppers = [dev[2] for dev in deviations]

        errors_lower = [(mean - lower) for mean, lower in zip(means, lowers)]
        errors_upper = [(upper - mean) for mean, upper in zip(means, uppers)]

        fig, ax = plt.subplots(figsize=config.figsize)
        x_pos = np.arange(len(names))

        ax.bar(
            x_pos,
            means,
            yerr=[errors_lower, errors_upper],
            capsize=5,
            alpha=0.7,
            ecolor="black",
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("Relative Neutrino Deviation")
        ax.set_title(
            f"Neutrino Reconstruction Deviation ({config.confidence*100:.0f}% CI)"
        )
        ax.set_ylim(0, None)
        ax.grid(axis="y", alpha=config.alpha)
        

        return fig, ax

    @staticmethod
    def plot_binned_deviation(
        bin_centers: np.ndarray,
        binned_deviations: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
        reconstructor_names: List[str],
        bin_counts: np.ndarray,
        bins: np.ndarray,
        feature_label: str,
        config: PlotConfig,
    ):
        """Plot binned neutrino deviation vs. a feature."""
        fig, ax = plt.subplots(figsize=config.figsize)
        color_map = plt.get_cmap("tab10")

        # Plot each reconstructor
        for index, (name, (mean_dev, lower, upper)) in enumerate(
            zip(reconstructor_names, binned_deviations)
        ):
            if config.show_errorbar:
                errors_lower = mean_dev - lower
                errors_upper = upper - mean_dev
                ax.errorbar(
                    bin_centers,
                    mean_dev,
                    yerr=[errors_lower, errors_upper],
                    fmt="x",
                    label=name,
                    color=color_map(index),
                    linestyle="None",
                )
            else:
                ax.plot(
                    bin_centers,
                    mean_dev,
                    label=name,
                    color=color_map(index),
                )

        # Configure main axes
        ax.set_xlabel(feature_label)
        ax.set_ylabel("Relative Neutrino Deviation")
        ax.set_ylim(0, None)
        ax.set_xlim(bins[0], bins[-1])
        ax.grid(alpha=config.alpha)
        ax.legend(loc="best")

        # Add event count histogram
        ax_twin = ax.twinx()
        ax_twin.bar(
            bin_centers,
            bin_counts,
            width=(bins[1] - bins[0]),
            alpha=0.2,
            color="red",
            label="Event Count",
        )
        ax_twin.set_ylabel("Event Count", color="red")
        ax_twin.tick_params(axis="y", labelcolor="red")

        # Set title
        title = f"Neutrino Deviation per Bin vs {feature_label}"
        if config.show_errorbar:
            title += f" ({config.confidence*100:.0f}% CI)"
        ax.set_title(title)

        
        return fig, ax

    @staticmethod
    def plot_component_deviation_histograms(
        predicted_neutrinos: np.ndarray,
        true_neutrinos: np.ndarray,
        reconstructor_names: List[str],
        event_weights: Optional[np.ndarray] = None,
        bins: int = 50,
        xlims: Optional[Tuple[float, float]] = None,
        figsize: Tuple[int, int] = (15, 10),
        component_labels: List[str] = ["$p_x$", "$p_y$", "$p_z$"],
    ):
        """
        Plot histograms of relative deviation for each neutrino momentum component.

        Args:
            predicted_neutrinos: List of predicted neutrino momenta arrays
                                 [(n_events, 2, 3), ...] for each reconstructor
            true_neutrinos: True neutrino momenta (n_events, 2, 3)
            reconstructor_names: List of reconstructor names
            event_weights: Optional event weights (n_events,)
            bins: Number of bins for histograms
            xlims: Optional x-axis limits (min, max)
            figsize: Figure size
            component_labels: Labels for momentum components

        Returns:
            Tuple of (figure, axes)
        """
        n_components = 3  # px, py, pz
        n_reconstructors = len(predicted_neutrinos)

        fig, axes = plt.subplots(2, n_components, figsize=figsize)
        color_map = plt.get_cmap("tab10")

        if event_weights is None:
            event_weights = np.ones(true_neutrinos.shape[0])

        # Extend weights for both neutrinos
        weights_extended = np.concatenate([event_weights, event_weights])

        for comp_idx in range(n_components):
            # Top row: First neutrino
            ax_top = axes[0, comp_idx]
            # Bottom row: Second neutrino (anti-neutrino)
            ax_bottom = axes[1, comp_idx]

            for recon_idx, (pred, name) in enumerate(
                zip(predicted_neutrinos, reconstructor_names)
            ):
                # Compute relative deviation for this component
                # First neutrino (neutrino)
                diff_nu1 = np.abs(pred[:, 0, comp_idx] - true_neutrinos[:, 0, comp_idx])
                rel_dev_nu1 = np.divide(
                    diff_nu1,
                    np.abs(true_neutrinos[:, 0, comp_idx]),
                    out=np.zeros_like(diff_nu1, dtype=float),
                    where=true_neutrinos[:, 0, comp_idx] != 0,
                )

                # Second neutrino (anti-neutrino)
                diff_nu2 = np.abs(pred[:, 1, comp_idx] - true_neutrinos[:, 1, comp_idx])
                rel_dev_nu2 = np.divide(
                    diff_nu2,
                    np.abs(true_neutrinos[:, 1, comp_idx]),
                    out=np.zeros_like(diff_nu2, dtype=float),
                    where=true_neutrinos[:, 1, comp_idx] != 0,
                )

                # Determine bins
                if xlims is not None:
                    bin_edges = np.linspace(xlims[0], xlims[1], bins + 1)
                else:
                    # Combine all data to get consistent binning
                    all_dev = np.concatenate([rel_dev_nu1, rel_dev_nu2])
                    bin_edges = np.linspace(
                        np.percentile(all_dev, 1), np.percentile(all_dev, 99), bins + 1
                    )

                # Plot histograms
                ax_top.hist(
                    rel_dev_nu1,
                    bins=bin_edges,
                    weights=event_weights,
                    alpha=0.5,
                    label=name,
                    color=color_map(recon_idx),
                    histtype="step",
                    linewidth=2,
                    density=True,
                )
                ax_top.axvline(
                    np.median(rel_dev_nu1), color=color_map(recon_idx), linestyle="--"
                )

                ax_bottom.hist(
                    rel_dev_nu2,
                    bins=bin_edges,
                    weights=event_weights,
                    alpha=0.5,
                    label=name,
                    color=color_map(recon_idx),
                    histtype="step",
                    linewidth=2,
                    density=True,
                )
                ax_bottom.axvline(
                    np.median(rel_dev_nu2), color=color_map(recon_idx), linestyle="--"
                )

            # Configure top axes
            ax_top.set_xlabel(f"Relative Deviation {component_labels[comp_idx]}")
            ax_top.set_ylabel("Density")
            ax_top.set_title(f"Neutrino {component_labels[comp_idx]}")
            ax_top.grid(alpha=0.3)
            ax_top.legend(loc="best")

            # Configure bottom axes
            ax_bottom.set_xlabel(f"Relative Deviation {component_labels[comp_idx]}")
            ax_bottom.set_ylabel("Density")
            ax_bottom.set_title(f"Antineutrino {component_labels[comp_idx]}")
            ax_bottom.grid(alpha=0.3)
            ax_bottom.legend(loc="best")

        fig.suptitle("Neutrino Component Relative Deviation Distribution", fontsize=14)
        
        return fig, axes

    @staticmethod
    def plot_overall_deviation_distribution(
        predicted_neutrinos: List[np.ndarray],
        true_neutrinos: np.ndarray,
        reconstructor_names: List[str],
        event_weights: Optional[np.ndarray] = None,
        bins: int = 50,
        xlims: Optional[Tuple[float, float]] = None,
        figsize: Tuple[int, int] = (12, 6),
    ):
        """
        Plot histogram of overall relative deviation distribution for all reconstructors.

        Args:
            predicted_neutrinos: List of predicted neutrino momenta arrays
                                 [(n_events, 2, 3), ...] for each reconstructor
            true_neutrinos: True neutrino momenta (n_events, 2, 3)
            reconstructor_names: List of reconstructor names
            event_weights: Optional event weights (n_events,)
            bins: Number of bins for histogram
            xlims: Optional x-axis limits (min, max)
            figsize: Figure size

        Returns:
            Tuple of (figure, axis)
        """
        fig, ax = plt.subplots(figsize=figsize)
        color_map = plt.get_cmap("tab10")

        if event_weights is None:
            event_weights = np.ones(true_neutrinos.shape[0])

        # Extend weights for both neutrinos
        weights_extended = np.concatenate([event_weights, event_weights])

        # Collect all deviations to determine consistent binning
        all_deviations = []

        for pred in predicted_neutrinos:
            # Compute L2 norm deviation for each neutrino
            diff_norm = np.linalg.norm(pred - true_neutrinos, axis=-1)
            true_norm = np.linalg.norm(true_neutrinos, axis=-1)
            rel_dev = np.divide(
                diff_norm,
                true_norm,
                out=np.zeros_like(diff_norm, dtype=float),
                where=true_norm != 0,
            )
            # Flatten to get all neutrinos (both neutrino and anti-neutrino)
            rel_dev_flat = rel_dev.flatten()
            all_deviations.append(rel_dev_flat)

        # Determine bins
        if xlims is not None:
            bin_edges = np.linspace(xlims[0], xlims[1], bins + 1)
        else:
            # Use percentiles across all data
            combined = np.concatenate(all_deviations)
            bin_edges = np.linspace(
                np.percentile(combined, 1), np.percentile(combined, 99), bins + 1
            )

        # Plot histograms for each reconstructor
        for idx, (rel_dev_flat, name) in enumerate(
            zip(all_deviations, reconstructor_names)
        ):
            ax.hist(
                rel_dev_flat,
                bins=bin_edges,
                weights=weights_extended,
                alpha=0.5,
                label=name,
                color=color_map(idx),
                histtype="step",
                linewidth=2,
                density=True,
            )

        # Configure axes
        ax.set_xlabel("Relative Deviation")
        ax.set_ylabel("Density")
        ax.set_title("Overall Neutrino Reconstruction Relative Deviation Distribution")
        ax.grid(alpha=0.3)
        ax.legend(loc="best")

        
        return fig, ax


class DistributionPlotter:
    """Handles plotting of general distributions."""

    @staticmethod
    def plot_feature_distributions(
        feature_values: np.ndarray,
        feature_label: str,
        event_weights: Optional[np.ndarray] = None,
        labels: Optional[List[str]] = None,
        ax: Optional[plt.Axes] = None,
        bins: int = 50,
        xlims: Optional[Tuple[float, float]] = None,
    ):
        """
        Plot histogram of a feature's distribution.

        Args:
            feature_values: Array of feature values (n_events,)
            feature_name: Name of the feature
            feature_label: Label for the x-axis
            event_weights: Optional event weights (n_events,)
            bins: Number of bins for histogram
            xlims: Optional x-axis limits (min, max)
            figsize: Figure size

        Returns:
            Tuple of (figure, axis)
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        if event_weights is None:
            event_weights = np.ones(len(feature_values[0]))

        # Determine bins
        if xlims is not None:
            bin_edges = np.linspace(xlims[0], xlims[1], bins + 1)
        else:
            # Combine all data to get consistent binning
            all_values = np.concatenate(feature_values)
            bin_edges = np.linspace(
                np.percentile(all_values, 1), np.percentile(all_values, 99), bins + 1
            )

        # Plot histograms
        for idx, values in enumerate(feature_values):
            label = labels[idx] if labels is not None else None

            # Filter out NaN and inf values
            valid_mask = np.isfinite(values)
            valid_values = values[valid_mask]
            valid_weights = event_weights[valid_mask]

            if len(valid_values) == 0:
                print(f"Warning: No valid values for {label}")
                continue

            ax.hist(
                valid_values,
                bins=bin_edges,
                weights=valid_weights,
                #alpha=0.5,
                label=label,
                histtype="step",
                linewidth=2,
                density=True,
                color=plt.get_cmap("tab10")(idx),
            )
        ax.set_xlabel(feature_label)
        ax.set_ylabel("Density")
        ax.grid(alpha=0.3)
        if labels is not None:
            ax.legend(loc="best")
        return ax
