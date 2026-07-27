import os
import sys
import traceback
from itertools import product
from pathlib import Path

import numpy as np


SRC_DIRECTORY = Path(__file__).resolve().parents[1]
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from pyVertexModel.algorithm.vertexModelBubbles import VertexModelBubbles
from pyVertexModel.parameters.set import PROJECT_DIRECTORY


ORIGINAL_WING_DISC_HEIGHT = 15.0
CELL_HEIGHTS = (
    np.array([0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0])
    * ORIGINAL_WING_DISC_HEIGHT
)
ASPECT_RATIOS = [1.0, 1.5, 2.5, 5.0, 10.0]
SHORT_AXIS = 4.0
SHARED_CYST_SOURCE = "Result/Cyst/cyst_scratch.tif"


def safe_float(value):
    return f"{value:g}".replace(".", "p")


def all_jobs():
    return list(product(range(len(ASPECT_RATIOS)), range(len(CELL_HEIGHTS))))


def parse_job_indices(argv):
    """
    Parse one parallel-array index, or explicit aspect/height indices.

    Supported forms:
      python run_cyst_aspect_cell_height.py 0
      python run_cyst_aspect_cell_height.py 2 4
      SLURM_ARRAY_TASK_ID=0 python run_cyst_aspect_cell_height.py
      PYVERTEXMODEL_ASPECT_INDEX=2 PYVERTEXMODEL_HEIGHT_INDEX=4 python ...
    """
    if len(argv) > 3:
        raise ValueError(
            "Expected one job index or two indices: "
            "aspect_index height_index"
        )

    if len(argv) == 3:
        aspect_idx = int(argv[1])
        height_idx = int(argv[2])
    elif (
        os.environ.get("PYVERTEXMODEL_ASPECT_INDEX") is not None
        or os.environ.get("PYVERTEXMODEL_HEIGHT_INDEX") is not None
    ):
        aspect_idx = int(os.environ.get("PYVERTEXMODEL_ASPECT_INDEX", "0"))
        height_idx = int(os.environ.get("PYVERTEXMODEL_HEIGHT_INDEX", "0"))
    else:
        if len(argv) == 2:
            job_idx_text = argv[1]
        else:
            job_idx_text = os.environ.get("SLURM_ARRAY_TASK_ID") or "0"

        job_idx = int(job_idx_text)
        jobs = all_jobs()
        if job_idx < 0 or job_idx >= len(jobs):
            raise ValueError(
                f"Job index must be between 0 and {len(jobs) - 1}, "
                f"got {job_idx}"
            )
        aspect_idx, height_idx = jobs[job_idx]

    if aspect_idx < 0 or aspect_idx >= len(ASPECT_RATIOS):
        raise ValueError(
            f"Aspect-ratio index must be between 0 and "
            f"{len(ASPECT_RATIOS) - 1}, got {aspect_idx}"
        )

    if height_idx < 0 or height_idx >= len(CELL_HEIGHTS):
        raise ValueError(
            f"Cell-height index must be between 0 and "
            f"{len(CELL_HEIGHTS) - 1}, got {height_idx}"
        )

    return aspect_idx, height_idx


def configure_model(aspect_ratio, cell_height):
    safe_ratio = safe_float(aspect_ratio)
    safe_height = safe_float(cell_height)

    v_model = VertexModelBubbles(
        create_output_folder=False,
        set_option="cyst_scratch",
    )

    v_model.set.export_images = False
    v_model.set.VTK = False
    v_model.set.force_cell_initialization = True
    v_model.set.CellHeight = float(cell_height)
    v_model.set.resize_z = None

    v_model.set.ellipsoid_axis1 = aspect_ratio * SHORT_AXIS
    v_model.set.ellipsoid_axis2 = SHORT_AXIS
    v_model.set.ellipsoid_axis3 = SHORT_AXIS

    v_model.set.lumen_axis1 = 0.5 * v_model.set.ellipsoid_axis1
    v_model.set.lumen_axis2 = 0.5 * v_model.set.ellipsoid_axis2
    v_model.set.lumen_axis3 = 0.5 * v_model.set.ellipsoid_axis3

    v_model.set.model_name = f"Cyst_AR_{safe_ratio}_H_{safe_height}"
    v_model.set.initial_filename_state = SHARED_CYST_SOURCE
    v_model.set.generated_initial_filename_state = (
        f"Result/Cyst/cyst_scratch_AR_{safe_ratio}_H_{safe_height}.tif"
    )
    v_model.set.OutputFolder = os.path.join(
        PROJECT_DIRECTORY,
        "Result",
        "aspect_ratio_cell_height_sweep",
        v_model.set.model_name,
    )

    v_model.set.update_derived_parameters()
    return v_model


def main():
    aspect_idx, height_idx = parse_job_indices(sys.argv)
    aspect_ratio = ASPECT_RATIOS[aspect_idx]
    cell_height = CELL_HEIGHTS[height_idx]
    job_idx = all_jobs().index((aspect_idx, height_idx))

    print("Python executable:", sys.executable)
    print("Job index:", job_idx)
    print("Aspect-ratio index:", aspect_idx)
    print("Cell-height index:", height_idx)
    print("Aspect ratio:", aspect_ratio)
    print("Cell height:", cell_height)

    shared_source = os.path.join(PROJECT_DIRECTORY, SHARED_CYST_SOURCE)
    if not os.path.exists(shared_source):
        raise FileNotFoundError(f"Shared cyst source file not found: {shared_source}")

    v_model = configure_model(aspect_ratio, cell_height)
    unique_generated_stem = os.path.join(
        PROJECT_DIRECTORY,
        v_model.set.generated_initial_filename_state,
    )

    print("Shared cyst source:", shared_source)
    print("Unique generated-state stem:", unique_generated_stem)
    print(
        "Final ellipsoid axes:",
        v_model.set.ellipsoid_axis1,
        v_model.set.ellipsoid_axis2,
        v_model.set.ellipsoid_axis3,
    )
    print(
        "Final lumen axes:",
        v_model.set.lumen_axis1,
        v_model.set.lumen_axis2,
        v_model.set.lumen_axis3,
    )
    print("Output folder:", v_model.set.OutputFolder)

    os.makedirs(v_model.set.OutputFolder, exist_ok=True)
    v_model.set.redirect_output()

    v_model.initialize()
    v_model.set.export_images = False
    print("Image export enabled:", v_model.set.export_images)
    print("VTK export enabled:", v_model.set.VTK)
    v_model.iterate_over_time()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
