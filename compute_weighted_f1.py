#!/usr/bin/env python3
"""Aggregate RPA/PA F1 scores with anomaly-count weights.

The script expects result summary CSVs in ``results/`` with ``RPA F1`` and ``PA F1``
entries, and the dataset index CSV in ``data/iops_competition/_index_iops_npz.csv``.
It updates the index file by adding (or refreshing) an ``anomaly_weight`` column
and prints the globally weighted F1 scores.
"""
from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
INDEX_CSV = BASE_DIR / "data" / "iops_competition" / "_index_iops_npz.csv"
WEIGHT_COLUMN = "anomaly_weight"
TARGET_PREFIX = "p2_"


@dataclass
class ScoreRow:
    dataset_key: str
    aff_f1: float
    rpa_f1: float
    pa_f1: float
    point_f1: float


def discover_dataset_key_from_name(name: str) -> str | None:
    """Extract the dataset key (matching index file stems) from a results name."""
    if TARGET_PREFIX in name:
        fragment = name.split(TARGET_PREFIX, 1)[1]
        fragment = fragment.replace("_summary", "")
        fragment = fragment.replace(".csv", "")
        return fragment
    return None


def load_result_scores(results_dir: Path) -> Dict[str, ScoreRow]:
    scores: Dict[str, ScoreRow] = {}
    for csv_path in sorted(results_dir.glob("*_summary.csv")):
        dataset_label: str | None = None
        aff_f1: float | None = None
        rpa_f1: float | None = None
        pa_f1: float | None = None
        point_f1: float | None = None

        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if len(row) < 2:
                    continue
                key = row[0].strip()
                value = row[1].strip()
                if not key:
                    continue
                if key == "Dataset":
                    dataset_label = value
                elif key == "Affiliation F1":
                    aff_f1 = float(value)
                elif key == "RPA F1":
                    rpa_f1 = float(value)
                elif key == "PA F1":
                    pa_f1 = float(value)
                elif key == "Point-wise F1":
                    point_f1 = float(value)

        if dataset_label is None:
            dataset_label = discover_dataset_key_from_name(csv_path.stem)
        if dataset_label is None:
            raise ValueError(f"Cannot determine dataset key for {csv_path}")

        dataset_key = dataset_label
        if dataset_key.startswith(TARGET_PREFIX):
            dataset_key = dataset_key[len(TARGET_PREFIX) :]
        dataset_key = dataset_key.strip()

        if None in (aff_f1, rpa_f1, pa_f1, point_f1):
            raise ValueError(f"Missing F1 scores in {csv_path}")

        scores[dataset_key] = ScoreRow(
            dataset_key=dataset_key,
            aff_f1=aff_f1,
            rpa_f1=rpa_f1,
            pa_f1=pa_f1,
            point_f1=point_f1,
        )

    if not scores:
        raise FileNotFoundError(f"No summary CSVs found in {results_dir}")
    return scores


def read_index_rows(index_csv: Path) -> Tuple[Iterable[Dict[str, str]], Iterable[str]]:
    with index_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Index file {index_csv} has no header")
        fieldnames = list(reader.fieldnames)
        rows = [dict(row) for row in reader]
    return rows, fieldnames


def write_index_rows(index_csv: Path, fieldnames: Iterable[str], rows: Iterable[Dict[str, str]]) -> None:
    with index_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def compute_weights(rows: Iterable[Dict[str, str]], p_column: str) -> Dict[str, float]:
    try:
        totals = [float(row[p_column]) for row in rows]
    except KeyError as exc:
        raise KeyError(f"Column '{p_column}' missing in index CSV") from exc

    total_p = sum(totals)
    if total_p <= 0:
        raise ValueError("Total anomaly count is non-positive; cannot compute weights")

    weights: Dict[str, float] = {}
    for row in rows:
        dataset_key = Path(row["file"]).stem
        weight = float(row[p_column]) / total_p
        weights[dataset_key] = weight
    return weights


def main() -> int:
    scores = load_result_scores(RESULTS_DIR)
    rows, fieldnames = read_index_rows(INDEX_CSV)

    weight_map = compute_weights(rows, p_column="p_test")

    weighted_aff = 0.0
    weighted_rpa = 0.0
    weighted_pa = 0.0
    weighted_point = 0.0
    missing_scores: list[str] = []

    for row in rows:
        dataset_key = Path(row["file"]).stem
        weight = weight_map.get(dataset_key, 0.0)
        if dataset_key not in scores:
            missing_scores.append(dataset_key)
            continue
        score_row = scores[dataset_key]
        weighted_aff += weight * score_row.aff_f1
        weighted_rpa += weight * score_row.rpa_f1
        weighted_pa += weight * score_row.pa_f1
        weighted_point += weight * score_row.point_f1
        row[WEIGHT_COLUMN] = f"{weight:.12f}"

    if missing_scores:
        names = ", ".join(sorted(set(missing_scores)))
        raise KeyError(
            "Missing result summaries for datasets in index: " + names
        )

    if WEIGHT_COLUMN not in fieldnames:
        fieldnames = list(fieldnames) + [WEIGHT_COLUMN]

    write_index_rows(INDEX_CSV, fieldnames, rows)

    print(f"Weighted Affiliation F1: {weighted_aff:.6f}")
    print(f"Weighted RPA F1: {weighted_rpa:.6f}")
    print(f"Weighted PA F1: {weighted_pa:.6f}")
    print(f"Weighted Point-wise F1: {weighted_point:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
