import os
import matplotlib.pyplot as plt
import numpy as np

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
        # Change this folder accordingly (it doesn't really matter the name of the folder, so feel free to rename it
        output_folder = os.path.join(PROJECT_DIRECTORY, 'Result/breaking/03-20_100221_noise_0.00e+00_bNoise_0.00e+00_lVol_1.00e+00_refV0_1.00e+00_kSubs_1.00e-01_lt_0.00e+00_refA0_9.00e-01_eARBarrier_8.00e-07_RemStiff_0.6_lS1_1.00e-01_lS2_1.00e-03_lS3_1.00e-01_ps_4.00e-05_lc_7.00e-05')

        # You can either load just one file or go through all of them
        #load_state(vModel, os.path.join(output_folder, 'data_step_0.89.pkl'))

        # Sort files by date
        all_files = os.listdir(output_folder)
        all_files = sorted(all_files, key=lambda x: os.path.getmtime(os.path.join(output_folder, x)))

        # Time here means just the number of intercalations or file number
        time = 0

        times = []

        # Energy for a specific cell. You can change the cell_id to any other cell you want to analyse until 149
        energy_lt_cells = []
        energy_surface_cells = []
        energy_volume_cells = []
        energy_tri_ar_cells = []
        neighbours_3d_cells = []
        neighbours_apical_cells = []
        neighbours_basal_cells = []
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

                # Initialize lists to store energies for the current file
                energy_lt_file = []
                energy_surface_file = []
                energy_volume_file = []
                energy_tri_ar_file = []
                neighbours_3d_file = []
                neighbours_apical_file = []
                neighbours_basal_file = []

                # Compute TriAR energy barrier
                kg_tri_ar = KgTriAREnergyBarrier(Geo)
                kg_tri_ar.compute_work(Geo, Set, None, False)

                # Compute the contractility
                kg_lt = KgContractility(Geo)
                kg_lt.compute_work(Geo, Set, None, False)

                # Compute Surface Tension
                kg_surface_area = KgSurfaceCellBasedAdhesion(Geo)
                kg_surface_area.compute_work(Geo, Set, None, False)

                # Compute Volume
                kg_volume = KgVolume(Geo)
                kg_volume.compute_work(Geo, Set, None, False)

                for cell_id in range(len(Geo.Cells)):
                    print('Cell: ', cell_id)
                    if Geo.Cells[cell_id].AliveStatus is None or cell_id in Geo.BorderCells:
                        continue

                    energy_lt_file.append(kg_lt.energy_per_cell[cell_id])
                    energy_surface_file.append(kg_surface_area.energy_per_cell[cell_id])
                    energy_volume_file.append(kg_volume.energy_per_cell[cell_id])
                    energy_tri_ar_file.append(kg_tri_ar.energy_per_cell[cell_id])

                    # Compute neighbours
                    neighbours_3d = Geo.Cells[cell_id].compute_neighbours()
                    neighbours_3d_file.append(neighbours_3d)
                    neighbours_apical = Geo.Cells[cell_id].compute_neighbours(location_filter='Top')
                    neighbours_apical_file.append(neighbours_apical)
                    neighbours_basal = Geo.Cells[cell_id].compute_neighbours(location_filter='Bottom')
                    neighbours_basal_file.append(neighbours_basal)

                # Append energies for the current file to the main lists
                energy_lt_cells.append(energy_lt_file)
                energy_surface_cells.append(energy_surface_file)
                energy_volume_cells.append(energy_volume_file)
                energy_tri_ar_cells.append(energy_tri_ar_file)
                neighbours_3d_cells.append(neighbours_3d_file)
                neighbours_apical_cells.append(neighbours_apical_file)
                neighbours_basal_cells.append(neighbours_basal_file)

        # Plot the energies and save it
        plt.figure()
        # The parameters shown in the legend are the ones used in the simulation
        plt.plot(times, [sum(energy) for energy in energy_lt_cells], label='Line tension %s' % Set.cLineTension)
        plt.plot(times, [sum(energy) for energy in energy_surface_cells],
                 label='Surface tension apical %s, basal %s, lateral %s' % (Set.lambdaS1, Set.lambdaS2, Set.lambdaS3))
        plt.plot(times, [sum(energy) for energy in energy_volume_cells], label='Volume %s' % Set.lambdaV)
        plt.plot(times, [sum(energy) for energy in energy_tri_ar_cells], label='TriAR energy barrier %s' % Set.lambdaR)
        plt.legend()
        # Save the plot
        plt.savefig(os.path.join(output_folder, 'total_energies.png'))

        # Plot the average neighbours and save it
        plt.figure()
        plt.plot(times, [np.mean(energy) for energy in neighbours_3d_cells], label='3D neighbours')
        plt.plot(times, [np.mean(energy) for energy in neighbours_apical_cells], label='Apical neighbours')
        plt.plot(times, [np.mean(energy) for energy in neighbours_basal_cells], label='Basal neighbours')
        plt.legend()
        # Save the plot
        plt.savefig(os.path.join(output_folder, 'average_neighbours.png'))

        # Plot the energies for a specific cell and save it
        for cell_id in range(len(Geo.Cells)):
            if Geo.Cells[cell_id].AliveStatus is None or cell_id in Geo.BorderCells:
                continue

            plt.figure()
            plt.plot(times, [energy[cell_id] for energy in energy_lt_cells], label='Line tension')
            plt.plot(times, [energy[cell_id] for energy in energy_surface_cells], label='Surface tension')
            plt.plot(times, [energy[cell_id] for energy in energy_volume_cells], label='Volume')
            plt.plot(times, [energy[cell_id] for energy in energy_tri_ar_cells], label='TriAR energy barrier')
            plt.legend()
            # Save the plot
            plt.savefig(os.path.join(output_folder, 'cell_%s_energies.png' % cell_id))

            plt.figure()
            plt.plot(times, [neighbours_3d[cell_id] for neighbours_3d in neighbours_3d_cells], label='3D neighbours')
            plt.plot(times, [neighbours_apical[cell_id] for neighbours_apical in neighbours_apical_cells], label='Apical neighbours')
            plt.plot(times, [neighbours_basal[cell_id] for neighbours_basal in neighbours_basal_cells], label='Basal neighbours')
            plt.legend()
            # Save the plot
            plt.savefig(os.path.join(output_folder, 'cell_%s_neighbours.png' % cell_id))

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

