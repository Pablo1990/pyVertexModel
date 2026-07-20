import os
import sys

from pyVertexModel.algorithm.vertexModelBubbles import VertexModelBubbles
from pyVertexModel.parameters.set import PROJECT_DIRECTORY


aspect_ratios = [1.0, 1.5, 2.5, 5.0, 10.0]

idx = int(sys.argv[1])

if idx < 0 or idx >= len(aspect_ratios):
    raise ValueError(
        f"Aspect-ratio index must be between 0 and "
        f"{len(aspect_ratios) - 1}, got {idx}"
    )

aspect_ratio = aspect_ratios[idx]
short_axis = 4.0
safe_ratio = str(aspect_ratio).replace(".", "p")


# 1. Load cyst_scratch defaults first
v_model = VertexModelBubbles(
    create_output_folder=False,
    set_option="cyst_scratch",
)


# 2. Override the geometry after loading the preset
v_model.set.ellipsoid_axis1 = aspect_ratio * short_axis
v_model.set.ellipsoid_axis2 = short_axis
v_model.set.ellipsoid_axis3 = short_axis

v_model.set.lumen_axis1 = 0.5 * v_model.set.ellipsoid_axis1
v_model.set.lumen_axis2 = 0.5 * v_model.set.ellipsoid_axis2
v_model.set.lumen_axis3 = 0.5 * v_model.set.ellipsoid_axis3


# 3. Give every run a unique model name and output folder
v_model.set.model_name = f"Cyst_AR_{safe_ratio}"

v_model.set.OutputFolder = os.path.join(
    PROJECT_DIRECTORY,
    "Result",
    "aspect_ratio_sweep",
    f"Cyst_AR_{safe_ratio}",
)


# 4. Recalculate anything derived from the settings
v_model.set.update_derived_parameters()


# 5. Confirm nothing was overwritten
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


# 6. Create the output folder / redirect logging
os.makedirs(v_model.set.OutputFolder, exist_ok=True)
v_model.set.redirect_output()


# 7. Run
v_model.initialize()
v_model.iterate_over_time()