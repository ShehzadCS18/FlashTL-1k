"""FlashTL-1K dataset statistics.

Reads the contextual metadata and saves dataset composition and descriptive
statistics to a single Excel workbook.
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd


CLASS_NAMES = {0: "Flashing Red", 1: "Flashing Yellow"}
CONDITION_ORDER = ["inside_day", "inside_night", "outside_day", "outside_night", "rain"]
CONDITION_NAMES = {
    "inside_day": "Inside-Day",
    "inside_night": "Inside-Night",
    "outside_day": "Outside-Day",
    "outside_night": "Outside-Night",
    "rain": "Rain",
}


def as_bool(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer, float, np.floating)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def infer_condition(row: pd.Series) -> str:
    if as_bool(row.get("clip_is_rain", False)):
        return "rain"
    viewpoint = "inside" if as_bool(row.get("clip_is_inside", False)) else "outside"
    illumination = "day" if as_bool(row.get("clip_is_day", False)) else "night"
    return f"{viewpoint}_{illumination}"


def build_composition_table(df: pd.DataFrame) -> pd.DataFrame:
    table = pd.crosstab(df["condition"], df["label"])
    table = table.reindex(index=CONDITION_ORDER, columns=[0, 1], fill_value=0)
    table.columns = [CLASS_NAMES[0], CLASS_NAMES[1]]
    table["Total"] = table.sum(axis=1)
    table.index = [CONDITION_NAMES[c] for c in table.index]

    total = pd.DataFrame(
        {
            CLASS_NAMES[0]: [int(table[CLASS_NAMES[0]].sum())],
            CLASS_NAMES[1]: [int(table[CLASS_NAMES[1]].sum())],
            "Total": [int(table["Total"].sum())],
        },
        index=["Total"],
    )
    return pd.concat([table, total])


def build_descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "fft_dominant_freq_hz",
        "fft_dominant_period_sec",
        "fft_flicker_ratio",
        "gdino_bbox_area_ratio",
        "depth_at_bbox",
        "depth_at_bbox_percentile",
    ]
    duty_columns = [c for c in df.columns if "duty" in c.lower()]
    columns = [c for c in preferred + duty_columns if c in df.columns]

    rows = []
    for label, class_name in CLASS_NAMES.items():
        subset = df[df["label"] == label]
        for column in columns:
            values = pd.to_numeric(subset[column], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append({
                "Class": class_name,
                "Feature": column,
                "Count": int(values.count()),
                "Mean": float(values.mean()),
                "Std": float(values.std()),
                "Median": float(values.median()),
                "Min": float(values.min()),
                "Max": float(values.max()),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate FlashTL-1K dataset statistics.")
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("dataset_statistics.xlsx"))
    args = parser.parse_args()

    df = pd.read_csv(args.context)
    required = {"label", "clip_is_rain", "clip_is_inside", "clip_is_day"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    df["condition"] = df.apply(infer_condition, axis=1)

    composition = build_composition_table(df)
    statistics = build_descriptive_statistics(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        composition.to_excel(writer, sheet_name="Composition")
        statistics.to_excel(writer, sheet_name="Descriptive Statistics", index=False)

    print(f"Saved dataset statistics to: {args.output}")


if __name__ == "__main__":
    main()
