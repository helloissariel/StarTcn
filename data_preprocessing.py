import os
from pathlib import Path

import numpy as np
import pandas as pd
from merlion.transform.normalize import MeanVarNormalize
from merlion.utils import TimeSeries
from sklearn.preprocessing import StandardScaler

from prepare_ucr_pretrain import build_dataset as build_ucr_dataset


# ===============
# Common helpers
# ===============

def norm(train, test):
    scaler = StandardScaler()
    scaler.fit(train)
    train_data = scaler.transform(train)
    test_data = scaler.transform(test)
    return train_data, test_data


def save_kpi_npz(output_path: Path, X_train, y_train, X_test, y_test, ts_train, ts_test):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        timestamp_train=ts_train,
        timestamp_test=ts_test,
    )


# =====================
# KPI / IoPS processing
# =====================

def iops_competition_preprocessing(
    train_csv: Path | str = Path("data/iops_competition/phase2_train.csv"),
    test_csv: Path | str = Path("data/iops_competition/phase2_test.csv"),
    output_dir: Path | str = Path("data/iops_competition/npz"),
    prefix: str = "p2_",
):
    train_csv = Path(train_csv)
    test_csv = Path(test_csv)
    output_dir = Path(output_dir)

    required_cols = {"timestamp", "value", "label", "KPI ID"}

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    missing_train = required_cols - set(train_df.columns)
    missing_test = required_cols - set(test_df.columns)
    if missing_train or missing_test:
        raise ValueError(
            f"Missing required columns. train missing={sorted(missing_train)}, test missing={sorted(missing_test)}"
        )

    train_df = train_df.rename(columns={"KPI ID": "kpi_id"})
    test_df = test_df.rename(columns={"KPI ID": "kpi_id"})

    train_df = train_df.astype({"timestamp": np.int64, "value": np.float64, "label": np.int64})
    test_df = test_df.astype({"timestamp": np.int64, "value": np.float64, "label": np.int64})

    kpi_ids = sorted(set(train_df["kpi_id"]) | set(test_df["kpi_id"]))
    for idx, kpi_id in enumerate(kpi_ids, start=1):
        train_subset = train_df[train_df["kpi_id"] == kpi_id].sort_values("timestamp")
        test_subset = test_df[test_df["kpi_id"] == kpi_id].sort_values("timestamp")

        if train_subset.empty or test_subset.empty:
            print(f"[WARN] KPI {kpi_id} missing train or test samples; skipping")
            continue

        X_train = train_subset["value"].to_numpy(dtype=np.float32).reshape(-1, 1)
        y_train = train_subset["label"].to_numpy(dtype=np.int64)
        X_test = test_subset["value"].to_numpy(dtype=np.float32).reshape(-1, 1)
        y_test = test_subset["label"].to_numpy(dtype=np.int64)
        ts_train = train_subset["timestamp"].to_numpy(dtype=np.int64)
        ts_test = test_subset["timestamp"].to_numpy(dtype=np.int64)

        safe_kpi = kpi_id.replace("-", "_")
        filename = f"{prefix}{idx:02d}_{safe_kpi}.npz"
        save_kpi_npz(output_dir / filename, X_train, y_train, X_test, y_test, ts_train, ts_test)
        print(f"Saved {filename}: train={len(X_train)}, test={len(X_test)}")


# ===========================
# UCR Anomaly (full dataset)
# ===========================

def ucr_anomaly_preprocessing(
    data_root: Path | str = Path("data/UCR_Anomaly_FullData"),
    output_path: Path | str = Path("data/UCR_Anomaly_FullData/ucr_anomaly_full.npz"),
    train_fraction: float = 0.6,
):
    data_root = Path(data_root)
    output_path = Path(output_path)

    if not data_root.exists():
        raise FileNotFoundError(f"UCR Anomaly directory not found: {data_root}")

    (
        X_train,
        y_train,
        X_test,
        y_test,
        train_segments,
        test_segments,
        series_ids,
    ) = build_ucr_dataset(data_root, train_fraction=train_fraction)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X_train=X_train.astype(np.float32),
        y_train=y_train.astype(np.int8),
        X_test=X_test.astype(np.float32),
        y_test=y_test.astype(np.int8),
        train_segments=train_segments,
        test_segments=test_segments,
        series_ids=np.array(series_ids, dtype=object),
        dataset_root=str(data_root.resolve()),
        train_fraction=np.float32(train_fraction),
    )
    print(
        f"Saved UCR Anomaly aggregate npz to {output_path}: "
        f"train={X_train.shape[0]}, test={X_test.shape[0]}"
    )


# ==================
# SWaT / WADI utils
# ==================

def swat_preprocessing():
    # the following code is adapted from the source code in [Zhihan Li et al. KDD21]
    # preprocess for SWaT. SWaT.A2_Dec2015, version 0
    dataset_folder = os.path.join("../data/", "swat")

    test_df = pd.read_excel(os.path.join(dataset_folder, "SWaT_Dataset_Attack_v0.xlsx"), header=1)

    test_df = test_df.set_index(" Timestamp")
    test_df["label"] = np.where(test_df["Normal/Attack"] == "Attack", 1, 0)
    test_df = test_df.drop("Normal/Attack", axis=1)
    assert test_df.shape == (449919, 52)

    train_df = pd.read_excel(os.path.join(dataset_folder, "SWaT_Dataset_Normal_v0.xlsx"), header=1)
    train_df = train_df.set_index(" Timestamp")
    train_df["label"] = np.where(train_df["Normal/Attack"] == "Attack", 1, 0)
    train_df = train_df.drop("Normal/Attack", axis=1)

    # following [Zhihan Li et al. KDD21] & [Dan Li. ICANN. 2019]
    # fow SWaT data, due to the cold start of the system, starting point is 21600
    train_df = train_df.iloc[21600:]
    assert train_df.shape == (475200, 52)

    output_dir = "../data/swat/"
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "SWaT_train.csv"))
    test_df.to_csv(os.path.join(output_dir, "SWaT_test.csv"))


def wadi_preprocessing():
    # preprocess for WADI. WADI.A2_19Nov2019
    dataset_folder = os.path.join("../data/", "wadi")

    train_df = pd.read_csv(os.path.join(dataset_folder, "WADI_14days_new.csv"), index_col=0, header=0)
    test_df = pd.read_csv(os.path.join(dataset_folder, "WADI_attackdataLABLE.csv"), index_col=0, header=1)

    train_df = train_df.iloc[:, 2:]
    test_df = test_df.iloc[:, 2:]

    train_df = train_df.fillna(train_df.mean(numeric_only=True))
    test_df = test_df.fillna(test_df.mean(numeric_only=True))
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)

    # trim column names
    train_df = train_df.rename(columns=lambda x: x.strip())
    test_df = test_df.rename(columns=lambda x: x.strip())

    train_df["label"] = np.zeros(len(train_df))
    test_df["label"] = np.where(test_df["Attack LABLE (1:No Attack, -1:Attack)"] == -1, 1, 0)
    test_df = test_df.drop(columns=["Attack LABLE (1:No Attack, -1:Attack)"])

    output_dir = "../data/wadi/"
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "WADI_train.csv"))
    test_df.to_csv(os.path.join(output_dir, "WADI_test.csv"))


def swat():
    dataset_folder = os.path.join("data/", "swat")
    train_df = pd.read_csv(os.path.join(dataset_folder, "SWaT_train.csv"))
    train_df = np.array(train_df.set_index(" Timestamp"))
    train_data = train_df[:, :51]
    test_df = pd.read_csv(os.path.join(dataset_folder, "SWaT_test.csv"))
    test_df = np.array(test_df.set_index(" Timestamp"))
    test_data = test_df[:, :51]

    train_data, test_data = norm(train_data, test_data)

    train_labels = train_df[:, 51]
    test_labels = test_df[:, 51]

    return train_data, test_data, train_labels, test_labels


def wadi():
    dataset_folder = os.path.join("data/", "wadi")
    train_df = pd.read_csv(os.path.join(dataset_folder, "WADI_train.csv"))
    train_df = np.array(train_df.set_index("Row"))
    train_data = train_df[:, :127]
    test_df = pd.read_csv(os.path.join(dataset_folder, "WADI_test.csv"))
    test_df = np.array(test_df.set_index("Row "))
    test_data = test_df[:, :127]

    train_data, test_data = norm(train_data, test_data)

    train_labels = train_df[:, 127]
    test_labels = test_df[:, 127]

    return train_data, test_data, train_labels, test_labels


# ==============================================
# Merlion datasets (IOpsCompetition, UCR, etc.)
# ==============================================

def other_datasets(time_series, meta_data):
    train_time_series_ts = TimeSeries.from_pd(time_series[meta_data.trainval])
    test_time_series_ts = TimeSeries.from_pd(time_series[~meta_data.trainval])
    train_labels = TimeSeries.from_pd(meta_data.anomaly[meta_data.trainval])
    test_labels = TimeSeries.from_pd(meta_data.anomaly[~meta_data.trainval])
    mvn = MeanVarNormalize()
    mvn.train(train_time_series_ts + test_time_series_ts)

    bias, scale = mvn.bias, mvn.scale
    bias, scale = list(bias.values())[0], list(scale.values())[0]

    train_time_series = train_time_series_ts.to_pd().to_numpy()
    train_data = (train_time_series - bias) / scale
    test_time_series = test_time_series_ts.to_pd().to_numpy()
    test_data = (test_time_series - bias) / scale

    train_labels = train_labels.to_pd().to_numpy()
    test_labels = test_labels.to_pd().to_numpy()

    return train_data, test_data, train_labels, test_labels


def other_datasets_no_Normalize(time_series, meta_data):
    train_time_series_ts = TimeSeries.from_pd(time_series[meta_data.trainval])
    test_time_series_ts = TimeSeries.from_pd(time_series[~meta_data.trainval])
    train_labels = TimeSeries.from_pd(meta_data.anomaly[meta_data.trainval])
    test_labels = TimeSeries.from_pd(meta_data.anomaly[~meta_data.trainval])

    train_data = train_time_series_ts.to_pd().to_numpy()
    test_data = test_time_series_ts.to_pd().to_numpy()

    train_labels = train_labels.to_pd().to_numpy()
    test_labels = test_labels.to_pd().to_numpy()

    return train_data, test_data, train_labels, test_labels


if __name__ == "__main__":
    iops_competition_preprocessing()
    ucr_anomaly_preprocessing()
