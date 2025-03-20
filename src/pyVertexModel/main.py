import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pyvista.core.utilities.cell_type_helper import cell_num

from src import PROJECT_DIRECTORY
from src.pyVertexModel.Kg.kgContractility import KgContractility
from src.pyVertexModel.Kg.kgSurfaceCellBasedAdhesion import KgSurfaceCellBasedAdhesion
from src.pyVertexModel.Kg.kgTriAREnergyBarrier import KgTriAREnergyBarrier
from src.pyVertexModel.Kg.kgVolume import KgVolume
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
        output_folder = os.path.join(PROJECT_DIRECTORY, 'Result/breaking/03-20_100319_noise_0.00e+00_bNoise_0.00e+00_lVol_1.00e+00_refV0_1.00e+00_kSubs_1.00e-01_lt_0.00e+00_refA0_9.00e-01_eARBarrier_8.00e-07_RemStiff_0.6_lS1_1.00e-01_lS2_1.00e-03_lS3_1.00e-01_ps_4.00e-05_lc_7.00e-05')

        # You can either load just one file or go through all of them
        #load_state(vModel, os.path.join(output_folder, 'data_step_0.89.pkl'))

        # Sort files by date
        all_files = os.listdir(output_folder)
        all_files = sorted(all_files, key=lambda x: os.path.getmtime(os.path.join(output_folder, x)))

        # Time here means just the number of intercalations or file number
        time = 0

        times = []
        energy_lt = []
        energy_surface = []
        energy_volume = []
        energy_tri_ar = []

        # Energy for a specific cell. You can change the cell_id to any other cell you want to analyse until 149
        cell_id = 0
        energy_lt_cell = []
        energy_surface_cell = []
        energy_volume_cell = []
        energy_tri_ar_cell = []
        for file_id, file in enumerate(all_files):
            if file.endswith('.pkl') and not file.__contains__(
                    'data_step_before_remodelling') and not file.__contains__('recoil'):
                # Load the state of the model
                load_state(vModel, os.path.join(output_folder, file))

                print('File: ', file)

                # Export images
                # vModel.set.export_images = True
                # temp_dir = os.path.join(vModel.set.OutputFolder, 'images')
                # screenshot(vModel, temp_dir)

                # Analyse the simulation
                all_cells, avg_cells = vModel.analyse_vertex_model()
                # Export excel with all_cells per file
                all_cells.to_excel(os.path.join(output_folder, 'all_cells_%s.xlsx' % file))

                Geo = vModel.geo
                Set = vModel.set
                Set.currentT = 0

                # Compute the contractility
                kg_lt = KgContractility(Geo)
                kg_lt.compute_work(Geo, Set, None, False)
                g_lt = kg_lt.g

                energy_lt_cell.append(kg_lt.energy_per_cell[cell_id])
                energy_lt.append(kg_lt.energy)

                # Compute Surface Tension
                kg_surface_area = KgSurfaceCellBasedAdhesion(Geo)
                kg_surface_area.compute_work(Geo, Set, None, False)
                g_surface = kg_surface_area.g

                energy_surface_cell.append(kg_lt.energy_per_cell[cell_id])
                energy_surface.append(kg_surface_area.energy)

                # Compute Volume
                kg_volume = KgVolume(Geo)
                kg_volume.compute_work(Geo, Set, None, False)
                g_volume = kg_volume.g

                energy_volume_cell.append(kg_volume.energy_per_cell[cell_id])
                energy_volume.append(kg_volume.energy)

                # Compute TriAR energy barrier
                kg_tri_ar = KgTriAREnergyBarrier(Geo)
                kg_tri_ar.compute_work(Geo, Set, None, False)
                g_tri_ar = kg_tri_ar.g

                energy_tri_ar_cell.append(kg_tri_ar.energy_per_cell[cell_id])
                energy_tri_ar.append(kg_tri_ar.energy)

                # Check for unreasonable geometries

                # Next time
                times.append(time)
                time += 1

        # Plot the energies and save it
        plt.figure()
        plt.plot(times, energy_lt, label='Line tension')
        plt.plot(times, energy_surface, label='Surface tension')
        plt.plot(times, energy_volume, label='Volume')
        plt.plot(times, energy_tri_ar, label='TriAR energy barrier')
        plt.legend()
        # Save the plot
        plt.savefig(os.path.join(output_folder, 'total_energies.png'))

        # Plot the energies for a specific cell and save it
        plt.figure()
        plt.plot(times, energy_lt_cell, label='Line tension')
        plt.plot(times, energy_surface_cell, label='Surface tension')
        plt.plot(times, energy_volume_cell, label='Volume')
        plt.plot(times, energy_tri_ar_cell, label='TriAR energy barrier')
        plt.legend()
        # Save the plot
        plt.savefig(os.path.join(output_folder, 'cell_%s_energies.png' % cell_id))

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

