import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from glob import glob
import argparse
import sys
import yaml

sys.path.append("../../")
import core  # Assuming core is a module in the project


def parse_model_name(model_name):
    """Extract hyperparameters from model name."""
    pattern = r"_d(\d+)_l(\d+)_h(\d+)"
    match = re.search(pattern, model_name)
    if match:
        return {
            "hidden_dim": int(match.group(1)),
            "num_layers": int(match.group(2)),
            "num_heads": int(match.group(3)),
        }
    return None


def load_and_evaluate_model(model_dir, validation_data, data_config):
    """Load a trained model and evaluate its performance."""
    model = core.reconstruction.KerasFFRecoBase(data_config, name="model")
    model.load_model(os.path.join(model_dir, "model.keras"))
    results = model.evaluate(validation_data)
    return results


def evaluate_all_models(models_dir, validation_data, data_config, model_type):
    """Evaluate all models in the specified directory."""
    evaluation_results = []
    model_dirs = glob(os.path.join(models_dir, f"{model_type}_*"))

    for model_dir in model_dirs:
        if "HLF" in model_dir and "HLF" not in model_type:
            continue  # Skip HLF models if not specified
        model_name = os.path.basename(model_dir)
        print(f"Evaluating model: {model_name}")
        results = load_and_evaluate_model(model_dir, validation_data, data_config)
        if results is not None:
            hyperparams = parse_model_name(model_name)
            if hyperparams is not None:
                results.update(hyperparams)
                results["model_name"] = model_name
                evaluation_results.append(results)

    return pd.DataFrame(evaluation_results)


def collect_results(models_dir, model_type="Raw_Transformer_Assignment"):
    """Collect training results from all model directories."""
    results = []
    # Find all model directories
    model_dirs = glob(os.path.join(models_dir, f"{model_type}_*"))

    print(f"Looking for {model_type} models in {models_dir}...")
    for model_dir in model_dirs:
        if "HLF" in model_dir and "HLF" not in model_type:
            continue  # Skip HLF models if not specified
        model_name = os.path.basename(model_dir)
        hyperparams = parse_model_name(model_name)
        print(f"Processing model: {model_name}")
        if hyperparams is None:
            continue

        # Load training history
        history_file = os.path.join(model_dir, f"history.npz")

        if not os.path.exists(history_file):
            print(f"Warning: History file not found for {model_name}")
            continue

        try:
            history = np.load(history_file)

            # Get best validation metrics
            val_loss = history["val_loss"]
            best_val_loss = np.min(val_loss)
            best_epoch = np.argmin(val_loss)

            # Get trainable parameters
            trainable_params = (
                int(history["trainable_params"])
                if "trainable_params" in history
                else None
            )

            # Try to get validation accuracy
            if "val_assignment_accuracy" in history:
                val_acc = history["val_assignment_accuracy"]
                best_val_acc = val_acc[best_epoch]
            elif "val_accuracy" in history:
                val_acc = history["val_accuracy"]
                best_val_acc = val_acc[best_epoch]
            else:
                best_val_acc = None

            result = {
                "model_name": model_name,
                "hidden_dim": hyperparams["hidden_dim"],
                "num_layers": hyperparams["num_layers"],
                "num_heads": hyperparams["num_heads"],
                "trainable_params": trainable_params,
                "best_val_loss": best_val_loss,
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
                "total_epochs": len(val_loss),
            }

            results.append(result)

        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            continue

    return pd.DataFrame(results)


def plot_grid_search_results(df, output_dir):
    """Create visualization of grid search results."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Heatmap of validation loss
    fig, ax = plt.subplots(figsize=(10, 8))
    pivot_loss = df.pivot_table(
        values="best_val_loss", index="num_layers", columns="hidden_dim", aggfunc="mean"
    )
    sns.heatmap(pivot_loss, annot=True, fmt=".4f", cmap="viridis_r")
    ax.set_title("Best Validation Loss by Hyperparameters")
    ax.set_ylabel("Transformer Stack Size")
    ax.set_xlabel("Embedding Dimension")
    
    fig.savefig(os.path.join(output_dir, "grid_search_loss_heatmap.pdf"), dpi=150)
    plt.close()

    # 2. Heatmap of validation accuracy (if available)
    if df["accuracy"].notna().any():
        fig, ax = plt.subplots(figsize=(10, 8))
        pivot_acc = df.pivot_table(
            values="accuracy",
            index="num_layers",
            columns="hidden_dim",
            aggfunc="mean",
        )
        sns.heatmap(pivot_acc, annot=True, fmt=".4f", cmap="viridis")
        ax.set_title("Best Validation Accuracy by Hyperparameters")
        ax.set_ylabel("Transformer Stack Size")
        ax.set_xlabel("Embedding Dimension")
        
        fig.savefig(
            os.path.join(output_dir, "grid_search_accuracy_heatmap.pdf"), dpi=150
        )
        plt.close()

    # 3. Heatmap of trainable parameters (if available)
    if df["trainable_params"].notna().any():
        fig, ax = plt.subplots(figsize=(10, 8))
        pivot_params = df.pivot_table(
            values="trainable_params",
            index="num_layers",
            columns="hidden_dim",
            aggfunc="mean",
        )
        sns.heatmap(pivot_params, annot=True, fmt=".0f", cmap="YlOrRd")
        ax.set_title("Trainable Parameters by Hyperparameters")
        ax.set_ylabel("Transformer Stack Size")
        ax.set_xlabel("Embedding Dimension")
        
        fig.savefig(os.path.join(output_dir, "grid_search_params_heatmap.pdf"), dpi=150)
        plt.close()

    # 6. Efficiency plot: Loss vs Parameters
    if df["trainable_params"].notna().any():

        # Create two subplots
        fig, ax2 = plt.subplots(1, figsize=(8, 6))

        # Plot 2: Accuracy vs Parameters (if available)
        if df["accuracy"].notna().any():
            scatter2 = ax2.scatter(
                df["trainable_params"] / 1e6,
                df["accuracy"],
                c=df["hidden_dim"],
                s=100,
                cmap="viridis",
                alpha=0.7,
                edgecolors="black",
            )
            ax2.set_xlabel("Trainable Parameters (Millions)")
            ax2.set_ylabel("Best Validation Accuracy")
            ax2.set_title("Model Efficiency: Accuracy vs Model Size")
            ax2.grid(True, alpha=0.3)
            fig.colorbar(scatter2, ax=ax2, label="Embedding Dimension")

        
        fig.savefig(os.path.join(output_dir, "grid_search_efficiency.pdf"), dpi=150)
        plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze Hyperparameter Grid Search Results"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="FeatureConcatTransformer",
        help="Type of model to analyze (default: FeatureConcatTransformer)",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/afs/desy.de/user/a/aulich/mva-trainer/models",
        help="Directory containing trained models (default: /afs/desy.de/user/a/aulich/mva-trainer/models)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_type = args.model_type
    root_dir = args.root_dir
    plt.rcParams.update({"font.size": 18})

    # Configuration
    ROOT_DIR = root_dir
    MODELS_DIR = ROOT_DIR
    OUTPUT_DIR = os.path.join(ROOT_DIR, "plots", "grid_search_analysis", model_type)
    CONFIG_DIR = os.path.join(ROOT_DIR, "../config")

    print(f"Collecting results from grid search (directory {ROOT_DIR})...")
    results_df = collect_results(MODELS_DIR, model_type=model_type)

    if len(results_df) == 0:
        print("No results found! Check your models directory.")
        return

    print(f"\nFound {len(results_df)} completed training runs")

    # Save results to CSV
    csv_path = os.path.join(OUTPUT_DIR, "grid_search_results.csv")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    # Load and evaluate models on validation data
    if "HLF" in model_type:
        data_config_file = os.path.join(CONFIG_DIR, "workspace_config_HLF.yaml")
    else:
        data_config_file = os.path.join(CONFIG_DIR, "nominal_load_config.yaml")

    data_config = core.get_load_config_from_yaml(data_config_file)
    data_loader = core.DataPreprocessor(data_config)

    with open(data_config_file, "r") as f:
        config_dict = yaml.safe_load(f)

    data_config = data_loader.load_from_npz(
        config_dict.data_path, max_events=4_000_000, event_numbers="odd"
    )

    validation_data,_ = data_loader.get_data()  # Get validation data

    print("\nEvaluating all models on validation data...")
    eval_results_df = evaluate_all_models(
        MODELS_DIR, validation_data, data_config=data_config, model_type=model_type
    )

    results_df.insert(
        len(results_df.columns),
        "accuracy",
        eval_results_df.set_index("model_name")
        .reindex(results_df["model_name"])["accuracy"]
        .values,
    )
    print(results_df)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("GRID SEARCH SUMMARY")
    print("=" * 60)
    print("\nBest model by validation loss:")
    best_model = results_df.loc[results_df["best_val_loss"].idxmin()]
    print(f"  Model: {best_model['model_name']}")
    print(f"  Hidden dim: {best_model['hidden_dim']}")
    print(f"  Num layers: {best_model['num_layers']}")
    print(f"  Val loss: {best_model['best_val_loss']:.6f}")
    if best_model["best_val_acc"] is not None:
        print(f"  Val accuracy: {best_model['best_val_acc']:.6f}")
    if best_model["trainable_params"] is not None:
        print(f"  Trainable params: {best_model['trainable_params']:,}")
    print(f"  Best epoch: {best_model['best_epoch']}")

    print("\nTop 5 models:")
    display_cols = [
        "hidden_dim",
        "num_layers",
        "trainable_params",
        "best_val_loss",
        "best_val_acc",
        "best_epoch",
    ]
    # Only include columns that exist and have data
    display_cols = [col for col in display_cols if col in results_df.columns]
    print(results_df.nsmallest(5, "best_val_loss")[display_cols].to_string(index=False))

    # Create visualizations
    print("\nCreating visualizations...")
    plot_grid_search_results(results_df, OUTPUT_DIR)
    print(f"Plots saved to {OUTPUT_DIR}")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
