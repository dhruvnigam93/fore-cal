import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.calibration import calibration_curve
import matplotlib.gridspec as gridspec
from typing import Union, List

from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from src.domain.Algorithms import TemperatureScaler, HistogramBinning, BBQ
from src.domain.entities import BinaryClassifier
from src.service.utils import test_dataset_binary_numeric, convert_to_binary_dataset

white_list_uci_ids = [2, 19, 20, 22, 26, 30, 69, 73, 75, 94, 107, 117, 159, 172, 222, 264, 267, 296, 327, 329, 350, 365,
                      372, 379, 380, 468, 545, 560, 563, 572, 579, 601, 603, 697, 827, 851, 863, 864, 880, 887, 890,
                      891,]


def calculate_ece(probs: Union[List, np.ndarray],
                  labels: Union[List, np.ndarray],
                  n_bins: int = 10) -> float:
    # Convert lists to numpy arrays if necessary
    probs = np.array(probs) if isinstance(probs, list) else probs
    labels = np.array(labels) if isinstance(labels, list) else labels

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs)

    for i in range(n_bins):
        # Binning by probability range
        bin_mask = (probs > bins[i]) & (probs <= bins[i + 1])
        if np.sum(bin_mask) > 0:
            bin_probs = probs[bin_mask]
            bin_labels = labels[bin_mask]

            bin_accuracy = np.mean(bin_labels)  # Fraction of positives in the bin
            bin_confidence = np.mean(bin_probs)  # Average predicted probability in the bin

            # Weighted absolute difference
            ece += np.abs(bin_confidence - bin_accuracy) * (len(bin_probs) / n)

    return ece


def create_bootstrap_dataset(df, n_bins=20, n_samples=100, random_state=None):
    """
    Creates a bootstrap dataset for calibration.

    Parameters:
    - df: DataFrame with columns 'predicted_prob' and 'label'.
    - n_bins: Number of bins to divide the predicted probabilities into.
    - n_samples: Number of bootstrap samples to generate per bin.
    - random_state: Random state for reproducibility.

    Returns:
    - DataFrame with bootstrapped mean predicted probabilities and actual instance rates.
    """
    # Set random state for reproducibility
    np.random.seed(random_state)

    df['bin'] = pd.cut(df['predicted_prob'], bins=n_bins, labels=False, include_lowest=True)

    # Prepare the bootstrap dataset
    bootstrap_data = []

    for bin_index in range(n_bins):

        bin_data = df[df['bin'] == bin_index]
        bin_size = len(bin_data)
        if bin_data.empty:
            continue

        for _ in range(n_samples):
            sample = bin_data.sample(n=len(bin_data), replace=True)
            bootstrap_data.append({f"{k}_mean": v for k, v in sample.drop(columns=["bin"]).mean().to_dict().items()} | {
                "bin_size": bin_size})
    bootstrap_df = pd.DataFrame(bootstrap_data)
    assert bootstrap_df.isna().sum().sum() == 0, "There are NaN values in the bootstrap dataset."

    return bootstrap_df


def classification_diagnostics(labels, predicted_probs, n_bins, plot=False) -> tuple:
    """

    :param labels:
    :param calibrated_probs:
    :param n_bins:
    :return:
    """
    ece = calculate_ece(predicted_probs, labels, n_bins=n_bins)
    auc = roc_auc_score(labels, predicted_probs)
    if plot:
        fig = plt.figure(figsize=(12, 12))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])  # The first plot is three times taller than the second

        # Create subplots in specified grid locations
        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1])

        # First subplot for the reliability diagram
        true_probas, pred_probas = calibration_curve(labels, predicted_probs, n_bins=n_bins)
        ax0.plot(pred_probas, true_probas, marker='o', linewidth=1, label='DNN Model')
        ax0.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')
        ax0.set_xlim([0, 1])
        ax0.set_ylim([0, 1])
        ax0.set_xlabel('Mean predicted probability')
        ax0.set_ylabel('Fraction of positives')
        ax0.set_title('Reliability Diagram')
        ax0.legend()
        ax0.grid(True)

        # Second subplot for the histogram of predicted probabilities
        ax1.hist(predicted_probs, bins=n_bins, range=(0, 1), alpha=0.75, color='blue')
        ax1.set_xlim([0, 1])
        ax1.set_xlabel('Predicted probabilities')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Predicted Probabilities')
        ax1.grid(True)

        # Show plot
        plt.tight_layout()
        plt.show()

    return None, ece, auc


def train_nn_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, plot=False):
    import matplotlib.pyplot as plt

    training_losses = []
    validation_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Average training loss
        train_loss /= len(train_loader)
        training_losses.append(train_loss)

        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()

        # Average validation loss
        val_loss /= len(val_loader)
        validation_losses.append(val_loss)

        # print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(training_losses, label='Training Loss')
        plt.plot(validation_losses, label='Validation Loss')
        plt.title('Training and Validation Loss Per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    return model


def is_valid_dataset(dataset_meta):
    is_binary = (dataset_meta['n_target_levels'] < 5)
    is_big = (dataset_meta['num_instances'] > 1000)
    return is_binary and is_big


def get_nn_model(X_train_, y_train_, X_val_, y_val_, hidden_size=100, hidden_layers=3, n_epochs=10):
    scaler = StandardScaler()
    X_train_ = scaler.fit_transform(X_train_)
    X_val_ = scaler.transform(X_val_)

    # Convert to PyTorch tensors
    X_train_ = torch.tensor(X_train_, dtype=torch.float32)
    y_train_ = torch.tensor(y_train_.values, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_, y_train_), batch_size=64, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val_, dtype=torch.float32), torch.tensor(y_val_.values, dtype=torch.float32)),
        batch_size=64, shuffle=False)

    model = BinaryClassifier(X_train_.shape[1], hidden_size, hidden_layers)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    model = train_nn_model(model, train_loader, val_loader, criterion, optimizer, n_epochs)

    return model, scaler


def get_nn_inference(model, X_val_, y_val_, scaler):
    X_val_ = scaler.transform(X_val_)
    val_data = TensorDataset(torch.tensor(X_val_, dtype=torch.float32),
                             torch.tensor(y_val_.values, dtype=torch.float32))
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

    model.eval()
    probs = []
    labels = []

    with torch.no_grad():
        for inputs, label in val_loader:
            output = model(inputs)
            probs.extend(output.squeeze().tolist())
            labels.extend(label.tolist())

    return probs


def get_results(input_tuple) -> dict:
    dataset_meta, model_params = input_tuple[0], input_tuple[1]
    print(
        f"Processing dataset: {dataset_meta['name']} with model parameters: {model_params}"
    )
    if is_valid_dataset(dataset_meta):
        print(f"Processing dataset: {dataset_meta['name']}")
        uri_id = dataset_meta["uci_id"]

        dataset_features, dataset_target = pickle.load(
            open(f"data/uci_datasets/uci_dataset_{uri_id}.pkl", "rb")
        )

        X, feat_names, y = convert_to_binary_dataset(dataset_features, dataset_target)
        test_dataset_binary_numeric(X, y)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.5
        )

        model, scaler = get_nn_model(
            X_train,
            y_train,
            X_val,
            y_val,
            model_params["base_nn_hidden_size"],
            model_params["base_nn_hidden_layers"],
            model_params["base_nn_n_epochs"],
        )
        probs = get_nn_inference(model, X_val, y_val, scaler)

        calib_df = pd.concat(
            [
                X_val.reset_index(drop=True),
                pd.DataFrame(probs, columns=["predicted_prob"]),
                pd.DataFrame(y_val.values, columns=["label"]),
            ],
            axis=1,
        )

        calib_train_df, calib_val_df = train_test_split(
            calib_df, test_size=0.4
        )

        calibration_models = []

        # # forecal plus based calibration
        # bootstrap_df = create_bootstrap_dataset(
        #     calib_train_df,
        #     model_params["calibration_forecal_n_bins"],
        #     model_params["calibration_forecal_n_samples"],
        #     random_state=42,
        # )

        # cols = list(
        #     bootstrap_df.columns
        # )  # make the predicted probability as the first column for monotonic constraint
        # cols.insert(0, cols.pop(cols.index("predicted_prob_mean")))
        # bootstrap_df = bootstrap_df.loc[:, cols]

        # X_train = bootstrap_df.drop(columns=["label_mean"])
        # y_train = bootstrap_df["label_mean"]

        # fc = RandomForestRegressor(
        #     n_estimators=100,
        #     random_state=42,
        #     monotonic_cst=([1] + [0] * (X_train.shape[1] - 1)),
        # )
        # fc.fit(X_train, y_train)
        # calibration_models.append(
        #     {"model": fc, "name": "forecal-plus", "features": X_train.columns.to_list()}
        # )

        # forecal based calibration

        bootstrap_df = create_bootstrap_dataset(
            calib_train_df,
            model_params["calibration_forecal_n_bins"],
            model_params["calibration_forecal_n_samples"],
            random_state=42,
        )
        cols = list(
            bootstrap_df.columns
        )  # make the predicted probability as the first column for monotonic constraint
        cols.insert(0, cols.pop(cols.index("predicted_prob_mean")))
        bootstrap_df = bootstrap_df.loc[:, cols]

        X_train = bootstrap_df[["predicted_prob_mean"]]
        y_train = bootstrap_df["label_mean"]
        weights = bootstrap_df["bin_size"]

        fc = RandomForestRegressor(
            n_estimators=model_params["calibration_forecal_n_estimators"],
            criterion=model_params["calibration_forecal_criterion"],
            random_state=42,
            monotonic_cst=[1],  # + [0] * (X_train.shape[1] - 1)),
        )
        fc.fit(
            X_train,
            y_train,
            sample_weight=weights
            if model_params["calibration_forecal_weight"]
            else None,
        )
        print(fc)
        calibration_models.append(
            {"model": fc, "name": "forecal", "features": ["predicted_prob_mean"]}
        )

        # isotonic regression based calibration
        X_train = calib_train_df["predicted_prob"].values.reshape(-1, 1)
        y_train = calib_train_df["label"].values
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(X_train.flatten(), y_train)
        calibration_models.append(
            {"model": ir, "name": "isotonic", "features": ["predicted_prob"]}
        )

        # logistic regression based calibration called platt scaling
        X_train = calib_train_df["predicted_prob"].values.reshape(-1, 1)
        y_train = calib_train_df["label"].values
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        calibration_models.append(
            {"model": lr, "name": "platt", "features": ["predicted_prob"]}
        )

        # # temperature scaling
        X_train = calib_train_df["predicted_prob"].values.reshape(-1, 1)
        y_train = calib_train_df["label"].values
        temp_scaler = TemperatureScaler()
        temp_scaler.fit(X_train, y_train)
        calibration_models.append(
            {"model": temp_scaler, "name": "tempscaler", "features": ["predicted_prob"]}
        )

        # # hist binning
        X_train = calib_train_df["predicted_prob"].values.reshape(-1, 1)
        y_train = calib_train_df["label"].values
        hist_binning = HistogramBinning(model_params["calibration_histbin_bins"])
        hist_binning.fit(X_train, y_train)
        calibration_models.append(
            {"model": hist_binning, "name": "hist-binning", "features": ["predicted_prob"]}
        )

        # # bbq
        X_train = calib_train_df["predicted_prob"].values.reshape(-1, 1)
        y_train = calib_train_df["label"].values
        bbq = BBQ()
        bbq.fit(X_train, y_train)
        calibration_models.append(
            {"model": bbq, "name": "bbq", "features": ["predicted_prob"]}
        )

        X_val = calib_val_df.drop(columns=["label"])
        y_val = calib_val_df["label"].values

        result_ls = []
        for models in [
                          {"model": None, "name": "baseline", "features": None}
                      ] + calibration_models:
            if models["name"] == "baseline":
                calibrated_probs = X_val["predicted_prob"]
            elif models["name"].startswith("fore"):
                X = X_val.rename(columns=lambda x: x + "_mean")[models["features"]]
                calibrated_probs = models["model"].predict(X)
            elif models["name"] == "platt":
                calibrated_probs = models["model"].predict_proba(
                    X_val[models["features"]]
                )[:, 1]
            elif models["name"] == "isotonic":
                calibrated_probs = models["model"].predict(X_val[models["features"]])
            elif models["name"] == "tempscaler":
                calibrated_probs = models["model"].predict(X_val["predicted_prob"])
            elif models["name"] in ["bbq", "hist-binning"]:
                calibrated_probs = models["model"].predict(X_val["predicted_prob"])
            else:
                raise ValueError(f"Unknown calibration model: {models['name']}")

            plt, ece, auc = classification_diagnostics(
                y_val, calibrated_probs, n_bins=10
            )
            result_ls.append({"name": models["name"], "ece": ece, "auc": auc})
            print(
                f"Expected Calibration Error (ECE) for {models['name']}: {ece:.4f} AUC: {auc:.4f}"
            )

        res = {
            "dataset": dataset_meta["name"],
            "results": result_ls,
            "uri_id": dataset_meta["uci_id"],
            "dataset_meta": dataset_meta,
            "model_params": model_params,
        }
        return res
    else:
        return None


def get_summary_metrics(in_df):
    ece_data = in_df[in_df["metric_name"] == "ece"]
    auc_data = in_df[in_df["metric_name"] == "auc"]
    algos_ = list(in_df["algo_name"].unique())

    # Aggregate the relative improvements for ECE and AUC
    ece_aggregate = (
            ece_data.groupby("algo_name")["metric_change"]
            .median()
            .to_frame(name="% Median change in ECE")
            * 100
    )
    auc_aggregate = (
            auc_data.groupby("algo_name")["metric_change"]
            .median()
            .to_frame(name="% Median change in AUC")
            * 100
    )

    # Function to calculate the percentage of datasets where each algorithm had the best value
    def calculate_best_percentage(df, metric_name):
        if metric_name == "ece":
            best_counts = (
                    df.pivot(
                        index="dataset_name",
                        columns="algo_name",
                        values="metric_value",
                    )[algos_]
                    .apply(lambda x: x.idxmin(), axis=1)
                    .value_counts(normalize=True)
                    * 100
            )
        elif metric_name == "auc":
            best_counts = (
                    df.pivot(
                        index="dataset_name",
                        columns="algo_name",
                        values="metric_value",
                    )[algos_]
                    .apply(lambda x: x.idxmax(), axis=1)
                    .value_counts(normalize=True)
                    * 100
            )
        return best_counts.to_frame(name="% Best " + metric_name.upper())

    return ece_aggregate.to_dict(), auc_aggregate.to_dict()


def convert_results_to_df(results_ls):
    rows = []
    for i, dataset_results in enumerate(results_ls):
        for entry in dataset_results["data"]:
            model_params = entry["model_params"]
            dataset_name = entry["dataset"]
            # run_params = entry["run_params"]
            for result in entry["results"]:
                algo_name = result["name"]
                for metric_name, metric_value in result.items():
                    if metric_name != "name":  # Exclude the algorithm name key from metrics
                        rows.append(
                            {
                                "run_id": i,
                                "dataset_name": dataset_name,
                                "dataset_size": entry["dataset_meta"]["num_instances"],
                                "algo_name": algo_name,
                                "metric_name": metric_name,
                                "metric_value": metric_value,
                            }
                            | model_params
                        )

    group_columns = ["run_id",  # for agg
                     "dataset_name", "dataset_size"]

    model_param_columns = list(results_ls[0]["model_params"].keys())

    grouped = (
        pd.DataFrame(rows).groupby(
            group_columns
            + model_param_columns
            + ["algo_name", "metric_name"],
            as_index=False,
        )
        .agg(
            metric_value=("metric_value", "mean"),
            observation_count=("metric_value", "size"),
        )
        .reset_index()
        .drop(columns=["index"])
    )
    # grouped.loc[grouped["metric_name"] == "auc", "metric_value"] = -grouped.loc[
    #     grouped["metric_name"] == "auc", "metric_value"
    # ]
    baseline = grouped.query("algo_name == 'baseline'")
    # Adding rank for each algorithm based on average metric value within each metric
    grouped["rank"] = grouped.groupby(
        group_columns + model_param_columns + ["metric_name"],
        as_index=False,
    )["metric_value"].rank(ascending=True)

    grouped = grouped.merge(
        baseline.drop(columns=["algo_name"]).rename(
            columns={"metric_value": "baseline_metric_value"}
        ),
        how="left",
        on=(group_columns + model_param_columns + ["metric_name"]),
    ).assign(
        metric_change=lambda x: (x["metric_value"] - x["baseline_metric_value"])
                                / x["baseline_metric_value"]
    )
    return grouped
