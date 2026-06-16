#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


KEYWORDS = [
    "env_route",
    "oracle_route_mae",
    "y_global_mae",
    "y_route_soft_mae",
    "y_route_final_mae",
    "alpha_mean",
    "alpha_std",
    "q_entropy",
    "q_max_mean",
    "q_oracle_entropy",
    "q_oracle_max_mean",
    "router_oracle_acc",
    "counts_per_head",
    "oracle_counts_per_head",
    "per_head_mae",
]


CANONICAL_KEYS = [
    "epoch",
    "step",
    "env_route/oracle_route_mae",
    "env_route/y_global_mae",
    "env_route/y_route_soft_mae",
    "env_route/y_route_final_mae",
    "env_route/alpha_mean",
    "env_route/alpha_std",
    "env_route/q_entropy",
    "env_route/q_max_mean",
    "env_route/q_oracle_entropy",
    "env_route/q_oracle_max_mean",
    "env_route/router_oracle_acc",
    "env_route/L_route_soft",
    "env_route/L_global",
    "env_route/L_final",
    "env_route/L_expert",
    "env_route/L_router_oracle",
    "env_route/L_balance",
    "env_route/L_diverse",
]


def flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        nk = f"{prefix}/{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, nk))
        else:
            out[nk] = v
    return out


def normalize_key(k: str) -> str:
    k = k.strip()
    k = k.replace("env_route.", "env_route/")
    k = k.replace("env_route_", "env_route/")
    return k


def extract_json_lines(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 直接 json line
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(flatten_dict(obj))
                    continue
            except Exception:
                pass

            # 日志中包含 {...}
            m = re.search(r"(\{.*\})", line)
            if m:
                try:
                    obj = json.loads(m.group(1))
                    if isinstance(obj, dict):
                        rows.append(flatten_dict(obj))
                except Exception:
                    pass
    return rows


def extract_key_values_from_text(path: Path) -> List[Dict[str, Any]]:
    """
    尝试从普通 train.log 中提取类似：
    env_route/alpha_mean=0.12
    env_route/q_max_mean: 0.5
    alpha_mean=...
    """
    rows = []
    pattern = re.compile(
        r"([A-Za-z0-9_/\.-]*(?:env_route|oracle_route|y_global|y_route|alpha|q_oracle|router_oracle|per_head|counts)[A-Za-z0-9_/\.-]*)\s*[:=]\s*([\-+]?\d+(?:\.\d+)?(?:e[\-+]?\d+)?)",
        re.IGNORECASE,
    )
    epoch_pattern = re.compile(r"(?:epoch|Epoch)\s*[:=\[\s]\s*(\d+)")

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not any(kw in line for kw in KEYWORDS):
                continue

            row = {}
            em = epoch_pattern.search(line)
            if em:
                row["epoch"] = int(em.group(1))

            for k, v in pattern.findall(line):
                nk = normalize_key(k)
                if not nk.startswith("env_route/"):
                    # 尽量把裸 key 归到 env_route 下
                    if any(x in nk for x in [
                        "oracle_route", "y_global", "y_route", "alpha",
                        "q_entropy", "q_max", "q_oracle",
                        "router_oracle", "per_head", "counts"
                    ]):
                        nk = f"env_route/{nk}"
                try:
                    row[nk] = float(v)
                except Exception:
                    pass

            if row:
                rows.append(row)

    return rows


def read_csv_file(path: Path) -> List[Dict[str, Any]]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return []

    cols = [str(c) for c in df.columns]
    if not any(any(kw in c for kw in KEYWORDS) for c in cols):
        return []

    df.columns = [normalize_key(str(c)) for c in df.columns]
    return df.to_dict("records")


def collect_rows(root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        name = path.name.lower()
        suffix = path.suffix.lower()

        file_rows: List[Dict[str, Any]] = []

        if suffix in [".csv"]:
            file_rows = read_csv_file(path)
        elif suffix in [".json", ".jsonl"]:
            file_rows = extract_json_lines(path)
        elif suffix in [".log", ".txt", ".out"]:
            file_rows = extract_json_lines(path)
            file_rows += extract_key_values_from_text(path)

        if file_rows:
            for r in file_rows:
                r["_source_file"] = str(path)
            rows.extend(file_rows)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # 统一一些可能的 key 名
    rename_map = {}
    for c in df.columns:
        nc = normalize_key(c)
        if nc != c:
            rename_map[c] = nc
    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def last_valid(df: pd.DataFrame, key: str) -> Optional[float]:
    if key not in df.columns:
        return None
    s = pd.to_numeric(df[key], errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.iloc[-1])


def best_min(df: pd.DataFrame, key: str) -> Optional[float]:
    if key not in df.columns:
        return None
    s = pd.to_numeric(df[key], errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.min())


def print_metric(df: pd.DataFrame, key: str):
    lv = last_valid(df, key)
    bv = best_min(df, key)
    if lv is None and bv is None:
        print(f"{key:36s}: MISSING")
    else:
        print(f"{key:36s}: last={lv}  best/min={bv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_dir",
        type=str,
        help="checkpoint run dir, e.g. checkpoints/.../stexpert_aligned_fpem_envroute_aux_k3",
    )
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--save_csv", type=str, default="")
    args = parser.parse_args()

    root = Path(args.run_dir)
    if not root.exists():
        raise FileNotFoundError(root)

    df = collect_rows(root)
    if df.empty:
        print("没有从该目录解析到 env_route 相关日志。")
        print("请确认日志中是否包含 env_route/xxx 字段，或者检查 train.py 是否真的写日志。")
        return

    # 尝试按 epoch/step 排序
    sort_cols = [c for c in ["epoch", "step", "global_step"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)

    if args.save_csv:
        out = Path(args.save_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"已保存解析结果: {out}")

    print("\n========== env-route metrics ==========")
    for key in [
        "env_route/oracle_route_mae",
        "env_route/y_global_mae",
        "env_route/y_route_soft_mae",
        "env_route/y_route_final_mae",
        "env_route/alpha_mean",
        "env_route/alpha_std",
        "env_route/q_entropy",
        "env_route/q_max_mean",
        "env_route/q_oracle_entropy",
        "env_route/q_oracle_max_mean",
        "env_route/router_oracle_acc",
        "env_route/L_route_soft",
        "env_route/L_global",
        "env_route/L_final",
        "env_route/L_expert",
        "env_route/L_router_oracle",
    ]:
        print_metric(df, key)

    print("\n========== diagnosis ==========")

    oracle = last_valid(df, "env_route/oracle_route_mae")
    y_global = last_valid(df, "env_route/y_global_mae")
    y_soft = last_valid(df, "env_route/y_route_soft_mae")
    y_final = last_valid(df, "env_route/y_route_final_mae")
    alpha = last_valid(df, "env_route/alpha_mean")
    qmax = last_valid(df, "env_route/q_max_mean")
    qo_max = last_valid(df, "env_route/q_oracle_max_mean")
    acc = last_valid(df, "env_route/router_oracle_acc")

    random_acc = 1.0 / args.k

    if oracle is not None and y_global is not None:
        if oracle < y_global:
            print(f"[OK] oracle_route_mae({oracle:.6f}) < y_global_mae({y_global:.6f})：多头有潜在上界。")
        else:
            print(f"[BAD] oracle_route_mae({oracle:.6f}) >= y_global_mae({y_global:.6f})：多头本身没有上界，先别看 router。")

    if y_soft is not None and oracle is not None:
        gap = y_soft - oracle
        print(f"[INFO] y_route_soft 与 oracle gap = {gap:.6f}，越小越说明 routed soft 学到 oracle 上界。")

    if y_final is not None and y_global is not None:
        diff = y_final - y_global
        print(f"[INFO] y_route_final - y_global = {diff:.6f}。若接近 0，说明 final 基本等于 global。")

    if alpha is not None:
        if alpha < 0.05:
            print(f"[WARN] alpha_mean={alpha:.6f}，很低，confidence fallback 基本退回全局不变头。")
        else:
            print(f"[OK] alpha_mean={alpha:.6f}，routed branch 已有一定参与。")

    if qmax is not None:
        print(f"[INFO] q_max_mean={qmax:.6f}；随机均匀大约是 {random_acc:.6f}。")

    if qo_max is not None:
        if qo_max <= random_acc + 0.05:
            print(f"[WARN] q_oracle_max_mean={qo_max:.6f} 接近均匀，说明 heads 还没明显分工。")
        else:
            print(f"[OK] q_oracle_max_mean={qo_max:.6f}，oracle 分配已有尖锐度。")

    if acc is not None:
        if acc <= random_acc:
            print(f"[WARN] router_oracle_acc={acc:.6f} <= random {random_acc:.6f}，router 还没学会选头。")
        else:
            print(f"[OK] router_oracle_acc={acc:.6f} > random {random_acc:.6f}，router 有效。")

    print("\n========== source files used ==========")
    for f in sorted(df["_source_file"].dropna().unique()):
        print(f)


if __name__ == "__main__":
    main()