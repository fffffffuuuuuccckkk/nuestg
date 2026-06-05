from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable

import numpy as np


INPUT_LEN = 12
OUTPUT_LEN = 12


DATASETS: Dict[str, Dict[str, object]] = {
    "newbike_chicago": {
        "official_name": "NewBike_Chicago",
        "year": "2019",
        "output_name": "NewBike_Chicago",
        "num_nodes": 609,
        "domain": "bike-sharing demand",
        "frequency_minutes": 60,
        "num_time_in_day": 24,
        "source_year": "2019",
        "ood_year_note": "following-year shift split from ST-OOD idx_shift.npy",
    },
    "taxi_chicago": {
        "official_name": "Taxi_Chicago",
        "year": "2013",
        "output_name": "Taxi_Chicago",
        "num_nodes": 77,
        "domain": "ride-hailing demand",
        "frequency_minutes": 60,
        "num_time_in_day": 24,
        "source_year": "2013",
        "ood_year_note": "following-year shift split from ST-OOD idx_shift.npy",
    },
    "speed_nyc": {
        "official_name": "Speed_NYC",
        "year": "2019",
        "output_name": "Speed_NYC",
        "num_nodes": 139,
        "domain": "traffic speed",
        "frequency_minutes": 10,
        "num_time_in_day": 144,
        "source_year": "2019",
        "ood_year_note": "following-year shift split from ST-OOD idx_shift.npy",
    },
    "pems08_ood": {
        "official_name": "PEMS08",
        "year": "2016",
        "output_name": "PEMS08-OOD",
        "num_nodes": 170,
        "domain": "traffic flow",
        "frequency_minutes": 5,
        "num_time_in_day": 288,
        "source_year": "2016",
        "ood_year_note": "held-out following-period shift split from ST-OOD idx_shift.npy",
        "use_shift_as_test": True,
    },
}


ALIASES = {
    "newbike": "newbike_chicago",
    "newbike_chicago": "newbike_chicago",
    "NewBike_Chicago": "newbike_chicago",
    "taxi": "taxi_chicago",
    "taxi_chicago": "taxi_chicago",
    "Taxi_Chicago": "taxi_chicago",
    "speed": "speed_nyc",
    "speed_nyc": "speed_nyc",
    "speed_NYC": "speed_nyc",
    "Speed_NYC": "speed_nyc",
    "pems08": "pems08_ood",
    "pems08_ood": "pems08_ood",
    "PEMS08": "pems08_ood",
    "PEMS08-OOD": "pems08_ood",
    "PEMS08_OOD": "pems08_ood",
}


def _resolve_names(names: Iterable[str]) -> list[str]:
    resolved = []
    for name in names:
        key = ALIASES.get(name, ALIASES.get(name.lower()))
        if key is None or key not in DATASETS:
            raise KeyError(f"Unknown ST-OOD dataset {name!r}; expected one of {sorted(DATASETS)}")
        if key not in resolved:
            resolved.append(key)
    return resolved


def _split_window(idx: np.ndarray, total_steps: int, input_len: int, output_len: int) -> tuple[int, int]:
    if idx.ndim != 1 or idx.size == 0:
        raise ValueError(f"idx must be a non-empty 1D array, got shape={idx.shape}")
    if not np.all(np.diff(idx) == 1):
        raise ValueError("BasicTS conversion expects contiguous ST-OOD idx arrays.")
    start = int(idx[0]) - input_len + 1
    end = int(idx[-1]) + output_len + 1
    if start < 0 or end > total_steps:
        raise ValueError(
            f"Invalid split window start={start}, end={end}, total_steps={total_steps}, "
            f"idx_range=({int(idx[0])}, {int(idx[-1])})"
        )
    return start, end


def _generate_timestamps(total_steps: int, frequency_minutes: int, start_time: str) -> np.ndarray:
    start = datetime.fromisoformat(start_time)
    values = np.zeros((total_steps, 2), dtype=np.float32)
    for step in range(total_steps):
        current = start + timedelta(minutes=frequency_minutes * step)
        minute_of_day = current.hour * 60 + current.minute
        values[step, 0] = minute_of_day / 1440.0
        values[step, 1] = current.weekday() / 7.0
    return values


def _load_raw_and_timestamps(npz_path: Path, spec: Dict[str, object]) -> tuple[np.ndarray, np.ndarray, float, float, str]:
    ptr = np.load(npz_path)
    data = np.asarray(ptr["data"], dtype=np.float32)
    if data.ndim != 3 or data.shape[-1] < 1:
        raise ValueError(f"{npz_path} data must be [T,N,C>=1], got {data.shape}")
    mean = float(np.asarray(ptr["mean"]).reshape(())) if "mean" in ptr.files else 0.0
    std = float(np.asarray(ptr["std"]).reshape(())) if "std" in ptr.files else 1.0
    values = data[..., 0] * std + mean
    if data.shape[-1] >= 3:
        timestamps = data[:, 0, 1:3].astype(np.float32)
        timestamp_source = "his.npz channels [time_of_day, day_of_week]"
    else:
        start_time = spec.get("start_time")
        if start_time is None:
            raise ValueError(
                f"{npz_path} does not contain timestamp channels and the dataset spec has no start_time. "
                "Set start_time and frequency_minutes to generate BasicTS timestamps."
            )
        timestamps = _generate_timestamps(data.shape[0], int(spec["frequency_minutes"]), str(start_time))
        timestamp_source = f"generated from start_time={start_time} and frequency_minutes={spec['frequency_minutes']}"
    if timestamps.ndim != 2 or timestamps.shape[1] != 2:
        raise ValueError(f"timestamps must be [T,2], got {timestamps.shape}")
    return values.astype(np.float32), timestamps, mean, std, timestamp_source


def convert_one(
    key: str,
    stood_root: Path,
    output_root: Path,
    input_len: int = INPUT_LEN,
    output_len: int = OUTPUT_LEN,
    overwrite: bool = True,
) -> Dict[str, object]:
    spec = DATASETS[key]
    official_name = str(spec["official_name"])
    year = str(spec["year"])
    source_dir = stood_root / "data" / official_name
    year_dir = source_dir / year
    npz_path = year_dir / "his.npz"
    adj_path = source_dir / "adj.npy"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    if not adj_path.exists():
        raise FileNotFoundError(adj_path)

    out_dir = output_root / str(spec["output_name"])
    out_dir.mkdir(parents=True, exist_ok=True)

    values, timestamps, source_mean, source_std, timestamp_source = _load_raw_and_timestamps(npz_path, spec)
    total_steps, num_nodes = values.shape
    expected_nodes = int(spec["num_nodes"])
    if num_nodes != expected_nodes:
        raise ValueError(f"{official_name} expected {expected_nodes} nodes, got {num_nodes}")

    split_meta = {}
    for split in ["train", "val", "test", "shift"]:
        idx_path = year_dir / f"idx_{split}.npy"
        idx = np.load(idx_path)
        start, end = _split_window(idx, total_steps, input_len, output_len)
        split_values = values[start:end].astype(np.float32)
        split_timestamps = timestamps[start:end].astype(np.float32)
        np.save(out_dir / f"{split}_data.npy", split_values)
        np.save(out_dir / f"{split}_timestamps.npy", split_timestamps)
        np.save(out_dir / f"st_ood_idx_{split}.npy", idx.astype(np.int64))
        expected_samples = int(idx.shape[0])
        actual_samples = int(split_values.shape[0] - input_len - output_len + 1)
        if actual_samples != expected_samples:
            raise AssertionError(
                f"{official_name}/{split}: expected {expected_samples} samples from ST-OOD idx, "
                f"got {actual_samples} after BasicTS conversion"
            )
        split_meta[split] = {
            "idx_file": str(idx_path),
            "idx_range": [int(idx[0]), int(idx[-1])],
            "source_window": [start, end],
            "data_shape": list(split_values.shape),
            "timestamps_shape": list(split_timestamps.shape),
            "basicts_num_samples": actual_samples,
        }

    if bool(spec.get("use_shift_as_test", False)):
        id_test_meta = dict(split_meta["test"])
        id_test_meta["evaluation_role"] = "in_distribution_reference_test"
        split_meta["id_test"] = id_test_meta
        for stem in ["data", "timestamps"]:
            src = out_dir / f"test_{stem}.npy"
            dst = out_dir / f"id_test_{stem}.npy"
            if src.exists():
                src.replace(dst)
            shift_src = out_dir / f"shift_{stem}.npy"
            np.save(src, np.load(shift_src, mmap_mode="r"))
        idx_src = out_dir / "st_ood_idx_test.npy"
        if idx_src.exists():
            idx_src.replace(out_dir / "st_ood_idx_id_test.npy")
        np.save(out_dir / "st_ood_idx_test.npy", np.load(out_dir / "st_ood_idx_shift.npy"))
        split_meta["test"] = dict(split_meta["shift"])
        split_meta["test"]["idx_file"] = str(year_dir / "idx_shift.npy")
        split_meta["test"]["evaluation_role"] = "out_of_distribution_shift_test"

    adj = np.load(adj_path).astype(np.float32)
    if adj.shape != (num_nodes, num_nodes):
        raise ValueError(f"{official_name} adj shape mismatch: expected {(num_nodes, num_nodes)}, got {adj.shape}")
    np.save(out_dir / "adj.npy", adj)
    with (out_dir / "adj_mx.pkl").open("wb") as f:
        pickle.dump(adj, f)

    meta = {
        "name": str(spec["output_name"]),
        "source": "ST-OOD",
        "source_root": str(stood_root),
        "official_name": official_name,
        "domain": str(spec["domain"]),
        "source_year": str(spec["source_year"]),
        "ood_year_note": str(spec["ood_year_note"]),
        "frequency_minutes": int(spec["frequency_minutes"]),
        "num_time_in_day": int(spec["num_time_in_day"]),
        "num_day_in_week": 7,
        "input_len": input_len,
        "output_len": output_len,
        "input_dim": 1,
        "output_dim": 1,
        "num_nodes": num_nodes,
        "num_time_steps_source": total_steps,
        "raw_value_description": "Unscaled first channel reconstructed as data[...,0] * his.npz['std'] + his.npz['mean'].",
        "source_normalization": {
            "mean": source_mean,
            "std": source_std,
            "note": "ST-OOD his.npz stores normalized target values; this conversion restores raw scale so the NUE-STG local runner fits its own train scaler.",
        },
        "timestamps_description": ["time_of_day", "day_of_week"],
        "timestamp_source": timestamp_source,
        "has_graph": True,
        "adjacency": {
            "source": str(adj_path),
            "files": ["adj.npy", "adj_mx.pkl"],
            "shape": list(adj.shape),
        },
        "split_policy": {
            "note": "Uses official ST-OOD idx_train/idx_val/idx_test/idx_shift arrays. Each BasicTS split stores the minimal contiguous time window needed to reproduce those samples.",
            "train": "official in-distribution training samples",
            "val": "official in-distribution validation samples",
            "test": (
                "official out-of-distribution shift samples"
                if bool(spec.get("use_shift_as_test", False))
                else "official in-distribution test samples"
            ),
            "id_test": (
                "official in-distribution test samples saved as id_test_* because test_* is reserved for OOD shift"
                if bool(spec.get("use_shift_as_test", False))
                else None
            ),
            "shift": "official out-of-distribution following-year test samples",
        },
        "splits": split_meta,
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, sort_keys=True)

    print(
        f"[prepared] {official_name} -> {out_dir} "
        f"train/val/test/shift samples="
        f"{split_meta['train']['basicts_num_samples']}/"
        f"{split_meta['val']['basicts_num_samples']}/"
        f"{split_meta['test']['basicts_num_samples']}/"
        f"{split_meta['shift']['basicts_num_samples']}"
    )
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ST-OOD his.npz datasets to BasicTS split format.")
    parser.add_argument(
        "--stood_root",
        default="/data/OuXiaoyu/mystg/datasets/ST-OOD",
        help="Root of the ST-OOD repository.",
    )
    parser.add_argument(
        "--output_root",
        default="/data/OuXiaoyu/mystg/datasets",
        help="Root where converted BasicTS dataset directories are written.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["newbike_chicago", "taxi_chicago", "speed_nyc"],
        help="Datasets to convert.",
    )
    parser.add_argument("--input_len", type=int, default=INPUT_LEN)
    parser.add_argument("--output_len", type=int, default=OUTPUT_LEN)
    args = parser.parse_args()

    stood_root = Path(args.stood_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    for key in _resolve_names(args.datasets):
        convert_one(key, stood_root, output_root, args.input_len, args.output_len)


if __name__ == "__main__":
    main()
