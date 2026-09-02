"""
Mockup Optuna pipeline for pre-ablation ellipsoid calibration.

This keeps the existing Optuna study/optimize settings from main_optuna.py,
but changes the objective to calibrate lambdaS1 == lambdaS3 and lambdaS2 for
each cyst ellipsoid aspect ratio before running ablation simulations.
"""

from __future__ import annotations

import os
import sys
import traceback
import argparse
from functools import partial
from pathlib import Path

import numpy as np
import optuna
import pandas as pd


SRC_DIRECTORY = Path(__file__).resolve().parents[1]
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from pyVertexModel import PROJECT_DIRECTORY
from pyVertexModel.algorithm.vertexModelBubbles import VertexModelBubbles
from pyVertexModel.util.space_exploration import plot_optuna_all


ASPECT_RATIOS = [1.0, 1.5, 2.5, 5.0, 10.0]
SHORT_AXIS = 4.0
SHARED_CYST_SOURCE = "Result/Cyst/cyst_scratch.tif"
NUM_TRIALS = 500
STUDY_PREFIX = "VertexModel_ellipsoid_stability"
RESULT_FOLDER = "Result/ellipsoid_surface_calibration"

# Keep model settings unchanged; only the scalar Optuna objective changes.
# The shape term prevents a low-gradient solution that has relaxed toward a
# sphere from being treated as mechanically calibrated for the target ellipsoid.
SHAPE_PENALTY_WEIGHT = 100.0
SIZE_PENALTY_WEIGHT = 100.0


def safe_float(value):
    return f"{value:g}".replace(".", "p")


def geometry_name(aspect_ratio):
    return f"Cyst_AR_{safe_float(aspect_ratio)}"


def create_ellipsoid_study_name(aspect_ratio):
    study_name = f"{STUDY_PREFIX}_{geometry_name(aspect_ratio)}"
    storage_name = "sqlite:///{}.db".format(
        os.path.join(PROJECT_DIRECTORY, "src", study_name)
    )
    return study_name, storage_name


def constant_volume_axes(aspect_ratio):
    long_axis = SHORT_AXIS * aspect_ratio ** (2 / 3)
    short_axis = SHORT_AXIS * aspect_ratio ** (-1 / 3)
    return long_axis, short_axis, short_axis


def configure_calibration_model(aspect_ratio, lambda_s13, lambda_s2):
    v_model = VertexModelBubbles(
        create_output_folder=False,
        set_option="cyst_scratch",
    )

    safe_ratio = safe_float(aspect_ratio)
    v_model.set.export_images = False
    v_model.set.VTK = False
    v_model.set.force_cell_initialization = True
    v_model.set.resize_z = None

    (
        v_model.set.ellipsoid_axis1,
        v_model.set.ellipsoid_axis2,
        v_model.set.ellipsoid_axis3,
    ) = constant_volume_axes(aspect_ratio)
    v_model.set.lumen_axis1 = 0.5 * v_model.set.ellipsoid_axis1
    v_model.set.lumen_axis2 = 0.5 * v_model.set.ellipsoid_axis2
    v_model.set.lumen_axis3 = 0.5 * v_model.set.ellipsoid_axis3

    v_model.set.model_name = geometry_name(aspect_ratio)
    v_model.set.initial_filename_state = SHARED_CYST_SOURCE
    v_model.set.generated_initial_filename_state = (
        f"Result/Cyst/cyst_scratch_AR_{safe_ratio}_calibration.tif"
    )

    v_model.set.lambdaS1 = lambda_s13
    v_model.set.lambdaS3 = lambda_s13
    v_model.set.lambdaS2 = lambda_s2

    v_model.set.update_derived_parameters()
    v_model.set.OutputFolder = None
    return v_model


def collect_alive_vertices(v_model):
    vertices = []
    for cell in v_model.geo.Cells:
        if cell is not None and cell.AliveStatus is not None and cell.Y is not None:
            vertices.append(cell.Y)

    if not vertices:
        return np.empty((0, 3))

    return np.concatenate(vertices, axis=0)


def measured_aspect_ratio(v_model):
    vertices = collect_alive_vertices(v_model)
    if vertices.size == 0:
        return np.nan

    spans = np.ptp(vertices, axis=0)
    minor_span = max(np.mean(spans[1:3]), np.finfo(float).eps)
    return spans[0] / minor_span


def measured_box_volume(v_model):
    vertices = collect_alive_vertices(v_model)
    if vertices.size == 0:
        return np.nan

    spans = np.ptp(vertices, axis=0)
    return float(np.prod(np.maximum(spans, np.finfo(float).eps)))


def ellipsoid_stability_objective(aspect_ratio, trial):
    lambda_s13 = trial.suggest_float("lambdaS1", 1e-7, 1, log=True)
    lambda_s2 = trial.suggest_float("lambdaS2", 1e-7, 1, log=True)

    v_model = configure_calibration_model(aspect_ratio, lambda_s13, lambda_s2)
    v_model.initialize()

    initial_aspect_ratio = measured_aspect_ratio(v_model)
    initial_box_volume = measured_box_volume(v_model)

    gr = v_model.single_iteration(post_operations=False)

    if not np.isfinite(gr):
        return np.inf

    relaxed_aspect_ratio = measured_aspect_ratio(v_model)
    relaxed_box_volume = measured_box_volume(v_model)
    relative_volume = relaxed_box_volume / initial_box_volume
    shape_error = (
        (relaxed_aspect_ratio - initial_aspect_ratio)
        / initial_aspect_ratio
    ) ** 2
    size_error = (np.log(max(relative_volume, np.finfo(float).eps))) ** 2

    trial.set_user_attr("initial_aspect_ratio", initial_aspect_ratio)
    trial.set_user_attr("initial_box_volume", initial_box_volume)
    trial.set_user_attr("aspect_ratio", aspect_ratio)
    trial.set_user_attr("relaxed_aspect_ratio", relaxed_aspect_ratio)
    trial.set_user_attr("relaxed_box_volume", relaxed_box_volume)
    trial.set_user_attr("relative_volume", relative_volume)
    trial.set_user_attr("gradient_norm", gr)
    trial.set_user_attr("shape_error", shape_error)
    trial.set_user_attr("size_error", size_error)

    return gr + SHAPE_PENALTY_WEIGHT * shape_error + SIZE_PENALTY_WEIGHT * size_error


def save_study_csvs(aspect_ratio, study, result_folder=None):
    if result_folder is None:
        result_folder = os.path.join(PROJECT_DIRECTORY, RESULT_FOLDER)

    os.makedirs(result_folder, exist_ok=True)

    trials_df = study.trials_dataframe(
        attrs=("number", "value", "params", "user_attrs", "state")
    )
    trials_path = os.path.join(
        result_folder,
        f"all_trials_AR_{safe_float(aspect_ratio)}.csv",
    )
    trials_df.to_csv(trials_path, index=False)

    best = study.best_trial
    best_results_df = pd.DataFrame([{
        "aspect_ratio": aspect_ratio,
        "lambdaS1": best.params["lambdaS1"],
        "lambdaS3": best.params["lambdaS1"],
        "lambdaS2": best.params["lambdaS2"],
        "objective": best.value,
        "gradient_norm": best.user_attrs["gradient_norm"],
        "initial_aspect_ratio": best.user_attrs["initial_aspect_ratio"],
        "relaxed_aspect_ratio": best.user_attrs["relaxed_aspect_ratio"],
        "relative_volume": best.user_attrs["relative_volume"],
        "shape_error": best.user_attrs["shape_error"],
        "size_error": best.user_attrs["size_error"],
    }])
    best_path = os.path.join(
        result_folder,
        f"best_parameters_AR_{safe_float(aspect_ratio)}.csv",
    )
    best_results_df.to_csv(best_path, index=False)

    print(f"Saved trials CSV: {trials_path}")
    print(f"Saved best-parameters CSV: {best_path}")

    return trials_df, best_results_df


def run_single_aspect_ratio(aspect_ratio, n_trials=NUM_TRIALS, result_folder=None):
    study = optuna.create_study(direction="minimize")
    study.optimize(
        partial(ellipsoid_stability_objective, aspect_ratio),
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1,
    )
    save_study_csvs(aspect_ratio, study, result_folder=result_folder)
    print("VALUE:", study.best_value)
    print("PARAMS:", study.best_params)
    print("ATTRS:", study.best_trial.user_attrs)
    return study


def main(aspect_ratio=None, n_trials=NUM_TRIALS):
    shared_source = os.path.join(PROJECT_DIRECTORY, SHARED_CYST_SOURCE)
    if not os.path.exists(shared_source):
        raise FileNotFoundError(f"Shared cyst source file not found: {shared_source}")

    result_folder = os.path.join(PROJECT_DIRECTORY, RESULT_FOLDER)
    os.makedirs(result_folder, exist_ok=True)

    if aspect_ratio is not None:
        run_single_aspect_ratio(
            aspect_ratio,
            n_trials=n_trials,
            result_folder=result_folder,
        )
        return

    best_results = []

    for aspect_ratio in ASPECT_RATIOS:
        study_name, storage_name = create_ellipsoid_study_name(aspect_ratio)
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            direction="minimize",
            load_if_exists=True,
        )

        if len(study.trials) < n_trials:
            remaining_trials = n_trials - len(study.trials)
            study.optimize(
                partial(ellipsoid_stability_objective, aspect_ratio),
                n_trials=remaining_trials,
                show_progress_bar=True,
                n_jobs=1,
            )

        save_study_csvs(aspect_ratio, study, result_folder=result_folder)

        best = study.best_trial
        best_results.append({
            "aspect_ratio": aspect_ratio,
            "lambdaS1": best.params["lambdaS1"],
            "lambdaS3": best.params["lambdaS1"],
            "lambdaS2": best.params["lambdaS2"],
            "objective": best.value,
            "gradient_norm": best.user_attrs["gradient_norm"],
            "initial_aspect_ratio": best.user_attrs["initial_aspect_ratio"],
            "relaxed_aspect_ratio": best.user_attrs["relaxed_aspect_ratio"],
            "relative_volume": best.user_attrs["relative_volume"],
            "shape_error": best.user_attrs["shape_error"],
            "size_error": best.user_attrs["size_error"],
        })

        print("Geometry:", geometry_name(aspect_ratio))
        print("Best parameters:", study.best_params)
        print("Best value:", study.best_value)
        print("Best trial:", study.best_trial)
        plot_optuna_all(
            result_folder,
            study_name,
            study,
        )

    best_results_df = pd.DataFrame(best_results)
    best_results_df.to_csv(
        os.path.join(result_folder, "best_parameters_by_AR.csv"),
        index=False,
    )
    print(best_results_df)


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--aspect-ratio", type=float)
        parser.add_argument("--trials", type=int, default=NUM_TRIALS)
        args = parser.parse_args()
        main(aspect_ratio=args.aspect_ratio, n_trials=args.trials)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
