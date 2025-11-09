#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
View a Hugging Face dataset saved with `save_to_disk` WITHOUT truncation.
- Shows dataset summary
- Prints first N or random N rows (full, no ellipsis)
- Optional: select columns, export to CSV, show schema, quick stats
"""

import argparse
from typing import List, Optional
from datasets import load_from_disk
import pandas as pd
import sys


def coerce_dataframe(
    ds_split, columns: Optional[List[str]], limit: int, sample: bool
) -> pd.DataFrame:
    # Validate/select columns
    if columns:
        valid_cols = [c for c in columns if c in ds_split.column_names]
        if not valid_cols:
            print(
                f"[WARN] None of the requested columns exist. Available: {ds_split.column_names}"
            )
            valid_cols = ds_split.column_names
    else:
        valid_cols = ds_split.column_names

    # Row subset (head or random sample)
    if sample:
        import random

        random.seed(0)
        k = min(limit, len(ds_split))
        idx = random.sample(range(len(ds_split)), k=k)
        subset = ds_split.select(idx)
    else:
        subset = ds_split.select(range(min(limit, len(ds_split))))

    # Proper column selection (do NOT use subset[cols], which means row indexing)
    subset = subset.select_columns(valid_cols)

    # Convert to pandas (no manual truncation here)
    df = subset.to_pandas()
    return df


def main():
    ap = argparse.ArgumentParser(
        description="Full, no-ellipsis viewer for a HF dataset saved to disk."
    )
    ap.add_argument(
        "--dataset_dir",
        required=True,
        help="Path to dataset dir (the folder containing dataset_dict.json).",
    )
    ap.add_argument(
        "--split",
        default="train",
        choices=["train", "valid", "test", "item_info"],
        help="Which split to view.",
    )
    ap.add_argument("--limit", type=int, default=10, help="How many rows to show.")
    ap.add_argument(
        "--columns",
        type=str,
        default="",
        help="Comma-separated list of columns to show. Leave empty for all columns.",
    )
    ap.add_argument(
        "--sample", action="store_true", help="Random sample instead of head."
    )
    ap.add_argument(
        "--export_csv", type=str, default="", help="Export shown rows to CSV."
    )
    ap.add_argument("--show_schema", action="store_true", help="Print features/schema.")
    ap.add_argument(
        "--stats",
        action="store_true",
        help="Print quick stats (unique users/items/timestamp range) if columns exist.",
    )
    args = ap.parse_args()

    ds = load_from_disk(args.dataset_dir)
    if args.split not in ds:
        print(f"[ERROR] Split '{args.split}' not found. Available: {list(ds.keys())}")
        sys.exit(1)

    d = ds[args.split]
    print(f"=== Dataset: {args.dataset_dir}")
    print(f"=== Split: {args.split} | Rows: {len(d)} | Columns: {d.column_names}")

    if args.show_schema:
        print("\n=== Schema / Features ===")
        print(d.features)

    cols = (
        [c.strip() for c in args.columns.split(",") if c.strip()]
        if args.columns
        else None
    )
    df = coerce_dataframe(d, cols, args.limit, args.sample)

    print("\n=== Preview (full, no truncation) ===")
    # Disable all pandas display truncation
    with pd.option_context(
        "display.max_colwidth",
        None,  # do not truncate cell content
        "display.max_rows",
        None,  # show all selected rows (limit controls size)
        "display.max_columns",
        None,  # show all columns
        "display.width",
        0,  # auto-wrap instead of truncating
    ):
        # to_string avoids '...' for long lists/strings
        print(df.to_string(index=False))

    if args.stats:
        print("\n=== Quick Stats ===")
        if "user_id" in d.column_names:
            try:
                print(f"unique users: {len(set(d['user_id']))}")
            except Exception:
                pass
        if "item_asin" in d.column_names:
            try:
                print(f"unique target items: {len(set(d['item_asin']))}")
            except Exception:
                pass
        if "item_id" in d.column_names:
            try:
                print(f"unique target item_ids: {len(set(d['item_id']))}")
            except Exception:
                pass
        if "timestamp" in d.column_names:
            try:
                ts = [t for t in d["timestamp"] if t is not None]
                if ts:
                    print(f"timestamp range: {min(ts)} ~ {max(ts)}")
            except Exception:
                pass

    if args.export_csv:
        out = args.export_csv
        df.to_csv(out, index=False)
        print(f"\n[Saved] {len(df)} rows → {out}")


if __name__ == "__main__":
    main()
