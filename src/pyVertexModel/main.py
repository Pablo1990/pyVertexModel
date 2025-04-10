import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
        output_folder = os.path.join(PROJECT_DIRECTORY, 'Result/03-20_100319')

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
        num_intercalations = []
        apical_flip_times = []
        basal_flip_times = []
        scutoid_cells = []
        for file_id, file in enumerate(all_files):
            if file.endswith('.pkl') and not file.__contains__(
                    'data_step_before_remodelling') and not file.__contains__('recoil'):
                # Load the state of the model
                load_state(vModel, os.path.join(output_folder, file))


                # Track intercalations
                if hasattr(vModel.geo, 'num_flips_this_step'):
                    num_intercalations.append(vModel.geo.num_flips_this_step)
                else:
                    num_intercalations.append(0)  # Default if not tracking flips

                # Track apical vs basal flips
                if hasattr(vModel.geo, 'apical_flips_this_step'):
                    apical_flip_times.append(vModel.geo.apical_flips_this_step)
                    basal_flip_times.append(vModel.geo.basal_flips_this_step)
                else:
                    apical_flip_times.append(0)
                    basal_flip_times.append(0)

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
                    if Geo.Cells[cell_id].AliveStatus is None:
                        continue

                    energy_lt_file.append(kg_lt.energy_per_cell[cell_id])
                    energy_surface_file.append(kg_surface_area.energy_per_cell[cell_id])
                    energy_volume_file.append(kg_volume.energy_per_cell[cell_id])
                    energy_tri_ar_file.append(kg_tri_ar.energy_per_cell[cell_id])

                    # Scutoid detection
                    apical_neigh = Geo.Cells[cell_id].compute_neighbours(location_filter='Top')
                    basal_neigh = Geo.Cells[cell_id].compute_neighbours(location_filter='Bottom')

                    if set(apical_neigh) != set(basal_neigh):
                        scutoid_cells.append({
                            'time': time,
                            'cell_id': cell_id,
                            'apical_neigh': len(apical_neigh),
                            'basal_neigh': len(basal_neigh)
                        })

                    # Compute neighbours
                    neighbours_3d = Geo.Cells[cell_id].compute_neighbours()
                    neighbours_3d_file.append(len(neighbours_3d))
                    neighbours_apical = Geo.Cells[cell_id].compute_neighbours(location_filter='Top')
                    neighbours_apical_file.append(len(neighbours_apical))
                    neighbours_basal = Geo.Cells[cell_id].compute_neighbours(location_filter='Bottom')
                    neighbours_basal_file.append(len(neighbours_basal))

                # Append energies for the current file to the main lists
                energy_lt_cells.append(energy_lt_file)
                energy_surface_cells.append(energy_surface_file)
                energy_volume_cells.append(energy_volume_file)
                energy_tri_ar_cells.append(energy_tri_ar_file)
                neighbours_3d_cells.append(neighbours_3d_file)
                neighbours_apical_cells.append(neighbours_apical_file)
                neighbours_basal_cells.append(neighbours_basal_file)

                times.append(time)
                time += 1

        # Custom cell geometry analysis
        weird_cells = []

        for cell_id in range(len(Geo.Cells)):
            if Geo.Cells[cell_id].AliveStatus is None:
                continue

            neighs = Geo.Cells[cell_id].compute_neighbours()
            if len(neighs) != 6:
                print(f'Cell {cell_id} has {len(neighs)} neighbours')
                print(f'Energy (LT): {kg_lt.energy_per_cell[cell_id]}')
                print(f'Energy (Surface): {kg_surface_area.energy_per_cell[cell_id]}')
                print(f'Energy (Volume): {kg_volume.energy_per_cell[cell_id]}')

                weird_cells.append({
                    'id': cell_id,
                    'neighs': len(neighs),
                    'energy_lt': kg_lt.energy_per_cell[cell_id],
                    'energy_surface': kg_surface_area.energy_per_cell[cell_id],
                    'energy_volume': kg_volume.energy_per_cell[cell_id]
                })
        # Save the weird cells to an excel sheet
        weird_cells_df = pd.DataFrame(weird_cells)
        weird_cells_df.to_excel(os.path.join(output_folder, 'weird_cells.xlsx'), index=False)

        pass

        # ====== PLOTTING SECTION (AFTER ALL DATA COLLECTION) ======

        # 1. Total Energies Plot
        plt.figure()
        plt.plot(times, [sum(energy) for energy in energy_lt_cells], label=f'Line tension {Set.cLineTension}')
        plt.plot(times, [sum(energy) for energy in energy_surface_cells],
                 label=f'Surface tension apical {Set.lambdaS1}, basal {Set.lambdaS2}, lateral {Set.lambdaS3}')
        plt.plot(times, [sum(energy) for energy in energy_volume_cells], label=f'Volume {Set.lambdaV}')
        plt.plot(times, [sum(energy) for energy in energy_tri_ar_cells], label=f'TriAR energy barrier {Set.lambdaR}')
        plt.legend()
        plt.savefig(os.path.join(output_folder, 'total_energies.png'))
        plt.close()

        # 2. Neighbor Analysis
        plt.figure(figsize=(12, 5))
        plt.subplot(121)
        plt.plot(times, [np.mean(n) for n in neighbours_3d_cells], label='3D neighbours')
        plt.plot(times, [np.mean(n) for n in neighbours_apical_cells], label='Apical neighbours')
        plt.plot(times, [np.mean(n) for n in neighbours_basal_cells], label='Basal neighbours')
        plt.legend()
        plt.title("Average Neighbors")

        plt.subplot(122)
        plt.plot(times, [np.std(n) for n in neighbours_3d_cells], label='3D neighbours')
        plt.plot(times, [np.std(n) for n in neighbours_apical_cells], label='Apical neighbours')
        plt.plot(times, [np.std(n) for n in neighbours_basal_cells], label='Basal neighbours')
        plt.legend()
        plt.title("Neighbor Variability")
        plt.savefig(os.path.join(output_folder, 'neighbour_analysis.png'))
        plt.close()

        # 3. Intercalation Analysis


        if len(num_intercalations) > 0:
            plt.figure(figsize=(15, 5))

            plt.subplot(131)
            plt.plot(times, num_intercalations, 'o-')
            plt.title("T1 Intercalation Events")
            plt.xlabel("Time")
            plt.ylabel("Count")

            plt.subplot(132)
            if len(apical_flip_times) == len(times):
                plt.plot(times, apical_flip_times, 'o-', label='Apical')
                plt.plot(times, basal_flip_times, 'o-', label='Basal')
                plt.legend()
                plt.title("Layer-Specific Flips")

            plt.subplot(133)
            plt.plot([sum(e) for e in energy_lt_cells], num_intercalations, 'o')
            plt.xlabel("Total Contractility Energy")
            plt.ylabel("Intercalations")
            plt.title("Energy vs Intercalations")

            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, 'intercalation_analysis.png'))
            plt.close()

        # 4. Energy Composition
        if len(energy_lt_cells) > 0:
            total_energy = np.array([sum(e) for e in energy_lt_cells]) + \
                           np.array([sum(e) for e in energy_surface_cells]) + \
                           np.array([sum(e) for e in energy_volume_cells]) + \
                           np.array([sum(e) for e in energy_tri_ar_cells])

            plt.figure()
            plt.stackplot(times,
                          [sum(e) / te for e, te in zip(energy_lt_cells, total_energy)],
                          [sum(e) / te for e, te in zip(energy_surface_cells, total_energy)],
                          [sum(e) / te for e, te in zip(energy_volume_cells, total_energy)],
                          [sum(e) / te for e, te in zip(energy_tri_ar_cells, total_energy)],
                          labels=['Line Tension', 'Surface', 'Volume', 'TriAR'])
            plt.legend(loc='upper left')
            plt.title("Energy Composition Over Time")
            plt.savefig(os.path.join(output_folder, 'energy_composition.png'))
            plt.close()

        # 5. Individual Cell Analysis (keep this in the loop but add proper closing)
        for cell_id in range(len(Geo.Cells)):
            if Geo.Cells[cell_id].AliveStatus is None or cell_id in Geo.BorderCells:
                continue

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

            # Energy plot
            ax1.plot(times, [e[cell_id] for e in energy_lt_cells], label='Line tension')
            ax1.plot(times, [e[cell_id] for e in energy_surface_cells], label='Surface')
            ax1.legend()

            # Neighbor plot
            ax2.plot(times, [n[cell_id] for n in neighbours_3d_cells], label='3D')
            ax2.plot(times, [n[cell_id] for n in neighbours_apical_cells], label='Apical')
            ax2.legend()

            plt.savefig(os.path.join(output_folder, f'cell_{cell_id}_analysis.png'))
            plt.close()


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

