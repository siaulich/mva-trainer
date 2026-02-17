import matplotlib.pyplot as plt
import numpy as np


def scale_axis_tick_labels(ax, scale=1e-3, axis='x', tick_format="{:.0f}"):
    """
    Scales the tick labels of a specified axis (x or y) by a given factor and formats them.

    Parameters:
    ax (matplotlib.axes.Axes): The Matplotlib axis to modify.
    scale (float): The factor by which to scale the axis ticks.
    axis (str): The axis to modify ('x' or 'y').
    tick_format (str): A format string for the tick labels (default is "{:.1f}").

    Returns:
    None
    """
    if axis == 'x':
        ticks = ax.get_xticks()
        xlims = ax.get_xlim()
        scaled_ticks = ticks * scale
        formatted_labels = [tick_format.format(tick) for tick in scaled_ticks]
        ax.set_xticks(ticks)
        ax.set_xticklabels(formatted_labels)
        ax.set_xlim(xlims)  # Ensure limits remain unchanged after scaling
    elif axis == 'y':
        ticks = ax.get_yticks()
        ylims = ax.get_ylim()
        scaled_ticks = ticks * scale
        formatted_labels = [tick_format.format(tick) for tick in scaled_ticks]
        ax.set_yticks(ticks)
        ax.set_yticklabels(formatted_labels)
        ax.set_ylim(ylims)  # Ensure limits remain unchanged after scaling
    else:
        raise ValueError("Axis must be 'x' or 'y'.")