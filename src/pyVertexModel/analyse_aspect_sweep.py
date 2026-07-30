#!/usr/bin/env python3

import os
import sys
import traceback
from pathlib import Path

import pandas as pd


SRC_DIRECTORY = Path(__file__).resolve().parents[1]
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from pyVertexModel import PROJECT_DIRECTORY
from pyVertexModel.analysis.analyse_simulation import _to_excel_or_csv, analyse_simulation


DEFAULT_SWEEP_FOLDER = (
    Path(PROJECT_DIRECTORY)
    / "Result"
    / "aspect_ratio_cell_height_ablation_sweep"
)
ANALYSIS_RESULTS_FOLDER = "analysis_results"


def parse_sweep_folder(argv):
    if len(argv) > 2:
        raise ValueError("Usage: analyse_aspect_sweep.py [sweep_results_folder]")

    folder = argv[1] if len(argv) == 2 else os.environ.get("PYVERTEXMODEL_SWEEP_FOLDER", DEFAULT_SWEEP_FOLDER)
    folder = Path(folder).expanduser()

    if not folder.is_absolute():
        folder = Path(PROJECT_DIRECTORY) / folder

    if not folder.exists():
        raise FileNotFoundError(f"Sweep folder not found: {folder}")

    return folder


def parse_run_metadata(folder_name):
    parts = folder_name.split("_")
    metadata = {"folder": folder_name}

    if "AR" in parts:
        metadata["aspect_ratio"] = float(parts[parts.index("AR") + 1].replace("p", "."))

    if "H" in parts:
        metadata["cell_height"] = float(parts[parts.index("H") + 1].replace("p", "."))

    metadata["ablation_case"] = parts[-1]
    return metadata


def iter_simulation_folders(sweep_folder):
    for path in sorted(sweep_folder.iterdir()):
        if path.is_dir() and path.name.startswith("Cyst_AR"):
            yield path


def main():
    sweep_folder = parse_sweep_folder(sys.argv)
    analysis_root = sweep_folder / ANALYSIS_RESULTS_FOLDER
    analysis_root.mkdir(parents=True, exist_ok=True)

    print(f"Sweep folder: {sweep_folder}", flush=True)
    print(f"Analysis results folder: {analysis_root}", flush=True)

    status_rows = []
    important_feature_rows = []

    for subfolder_path in iter_simulation_folders(sweep_folder):
        print(f"\nAnalyzing {subfolder_path.name}...", flush=True)

        status_row = parse_run_metadata(subfolder_path.name)
        status_row["path"] = str(subfolder_path)
        run_analysis_folder = analysis_root / subfolder_path.name

        try:
            (
                features_per_time_df,
                post_wound_features,
                important_features,
                features_per_time_all_cells_df,
            ) = analyse_simulation(str(subfolder_path), output_folder=str(run_analysis_folder))
        except Exception:
            status_row["status"] = "failed"
            status_row["error"] = traceback.format_exc()
            status_rows.append(status_row)
            print("Analysis failed", flush=True)
            _to_excel_or_csv(pd.DataFrame(status_rows), os.path.join(analysis_root, "analysis_status.xlsx"))
            continue

        if features_per_time_df is None:
            status_row["status"] = "skipped_no_output"
            status_rows.append(status_row)
            print("No analysis output produced for this folder.", flush=True)
            _to_excel_or_csv(pd.DataFrame(status_rows), os.path.join(analysis_root, "analysis_status.xlsx"))
            continue

        status_row["status"] = "analysed"
        status_row["num_time_points"] = len(features_per_time_df)
        status_row["num_post_wound_time_points"] = len(post_wound_features)
        status_rows.append(status_row)

        important_row = dict(status_row)
        if important_features is not None:
            important_row.update(important_features)
        important_feature_rows.append(important_row)

        print("Analysis finished", flush=True)
        print("Number of analysed time points:", len(features_per_time_df), flush=True)
        print("Number of post-wound time points:", len(post_wound_features), flush=True)

        _to_excel_or_csv(pd.DataFrame(status_rows), os.path.join(analysis_root, "analysis_status.xlsx"))
        _to_excel_or_csv(
            pd.DataFrame(important_feature_rows),
            os.path.join(analysis_root, "important_features_all_runs.xlsx"),
        )

    _to_excel_or_csv(pd.DataFrame(status_rows), os.path.join(analysis_root, "analysis_status.xlsx"))

    if important_feature_rows:
        important_features_df = pd.DataFrame(important_feature_rows)
        sort_columns = [
            column
            for column in ("aspect_ratio", "cell_height", "ablation_case")
            if column in important_features_df.columns
        ]
        if sort_columns:
            important_features_df = important_features_df.sort_values(sort_columns)

        _to_excel_or_csv(
            important_features_df,
            os.path.join(analysis_root, "important_features_all_runs.xlsx"),
        )

    print("\nSweep analysis complete.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
