import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from src import PROJECT_DIRECTORY
from src.pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage import VertexModelVoronoiFromTimeImage
from src.pyVertexModel.analysis.analyse_simulation import analyse_simulation
from src.pyVertexModel.util.utils import load_state

start_new = False
if start_new:
    vModel = VertexModelVoronoiFromTimeImage()
    vModel.initialize()
    vModel.iterate_over_time()
    analyse_simulation(vModel.set.OutputFolder)
else:
    debugging = True
    vModel = VertexModelVoronoiFromTimeImage(create_output_folder=False)

    if debugging:
        output_folder = os.path.join(PROJECT_DIRECTORY, 'Result/final_results/60_mins_Rok/')

        # Find the most recent simulation state file
        name_last_pkl_file = sorted(
            [f for f in os.listdir(output_folder) if
             f.endswith('.pkl') and not 'before_remodelling' in f and f.startswith('data_step_')],
            key=lambda x: os.path.getmtime(os.path.join(output_folder, x))
        )[-1]

        load_state(vModel, os.path.join(output_folder, name_last_pkl_file))
        vModel.set.OutputFolder = PROJECT_DIRECTORY + 'Result/final_results/60_mins_Rok/'
        os.makedirs(vModel.set.OutputFolder + '/images', exist_ok=True)

        vModel.reset_noisy_parameters()
        vModel.iterate_over_time()


        ## 🔹 Add Debugging Functions
        import matplotlib.pyplot as plt


        def plot_contractility(vModel):
            time_steps = list(range(len(vModel.contractility)))
            plt.plot(time_steps, vModel.contractility, label="Contractility Over Time")
            plt.xlabel("Time Step")
            plt.ylabel("Contractility")
            plt.title("Contractility Evolution During Wound Healing")
            plt.legend()
            plt.savefig(vModel.set.OutputFolder + "/contractility_plot.png")  # Save for meeting
            plt.show()


        plot_contractility(vModel)


        def plot_cell_volumes(vModel):
            time_steps = np.arange(len(vModel.cell_volumes))
            total_cell_volumes = [sum(cell_volumes) for cell_volumes in vModel.cell_volumes]

            plt.figure(figsize=(8, 5))
            plt.plot(time_steps, total_cell_volumes, label="Total Cell Volume", color='blue')
            plt.xlabel("Time Step")
            plt.ylabel("Total Cell Volume")
            plt.title("Volume Evolution During Remodeling")
            plt.legend()
            plt.grid()
            plt.show()


        def plot_surface_areas(vModel):
            time_steps = np.arange(len(vModel.surface_areas))
            total_surface_areas = [sum(surface_areas) for surface_areas in vModel.surface_areas]

            plt.figure(figsize=(8, 5))
            plt.plot(time_steps, total_surface_areas, label="Total Surface Area", color='red')
            plt.xlabel("Time Step")
            plt.ylabel("Total Surface Area")
            plt.title("Surface Area Evolution During Remodeling")
            plt.legend()
            plt.grid()
            plt.show()


        def plot_tetrahedral_quality(vModel):
            """Checks if tetrahedra are becoming degenerate over time."""
            detJ = np.array(vModel.tetrahedron_determinants)

            plt.figure(figsize=(8, 5))
            plt.hist(detJ, bins=50, color='purple', alpha=0.7)
            plt.xlabel("Determinant of Tetrahedra")
            plt.ylabel("Frequency")
            plt.title("Tetrahedral Quality Distribution")
            plt.grid()
            plt.show()


        ## 🔹 Call Debugging Functions
        plot_cell_volumes(vModel)
        plot_surface_areas(vModel)
        plot_tetrahedral_quality(vModel)

    else:
        load_state(vModel, os.path.join(PROJECT_DIRECTORY, 'Result/new_reference/before_ablation.pkl'))
        vModel.set.wing_disc()
        vModel.set.wound_default()
        vModel.set.dt0 = 0.1 * vModel.set.dt0  # Reduce time step by factor of 10
        vModel.set.dt = None
        vModel.set.OutputFolder = None
        vModel.set.update_derived_parameters()
        vModel.reset_noisy_parameters()
        vModel.set.redirect_output()
        vModel.iterate_over_time()

