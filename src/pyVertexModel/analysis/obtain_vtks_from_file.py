import os
import numpy as np

from pyVertexModel.Kg.kgSurfaceCellBasedAdhesion import KgSurfaceCellBasedAdhesion
from pyVertexModel.Kg.kgVolume import KgVolume
from pyVertexModel.algorithm.vertexModel import VertexModel
from pyVertexModel.analysis.analyse_simulation import create_video
from pyVertexModel.util.utils import load_state, screenshot

all_files = False
if all_files:
    # folders = [
    #     '/media/pablo/d7c61090-024c-469a-930c-f5ada47fb049/PabloVicenteMunuera/VertexModel/pyVertexModel/Result/final_results_wing_disc_real_bottom_right/',
    #     '/media/pablo/d7c61090-024c-469a-930c-f5ada47fb049/PabloVicenteMunuera/VertexModel/pyVertexModel/Result/final_results_wing_disc_real_top_right/',
    #     '/media/pablo/d7c61090-024c-469a-930c-f5ada47fb049/PabloVicenteMunuera/VertexModel/pyVertexModel/Result/final_results_wing_disc_real/',
    #     '/media/pablo/d7c61090-024c-469a-930c-f5ada47fb049/PabloVicenteMunuera/VertexModel/pyVertexModel/Result/final_results_wing_disc_real_bottom_left/',
    #        ]
    folders = [
        '/media/pablo/d7c61090-024c-469a-930c-f5ada47fb049/PabloVicenteMunuera/VertexModel/pyVertexModel/Result/to_analyse/'
    ]
    for folder in folders:
        # Get all the directories in the folder
        lst = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
        # Sort by name
        lst.sort()
        for c_dir in lst:
            print(c_dir)
            # if file is a directory
            if os.path.isdir(os.path.join(folder, c_dir)):
                if os.path.exists(os.path.join(folder, c_dir, 'images', 'combined_video.mp4')):
                    print('Video already exists for ' + c_dir)
                    continue
                vModel = VertexModel(create_output_folder=False)
                file_list = os.listdir(os.path.join(folder, c_dir))
                file_list.sort(reverse=True)  # Sort files by name
                for file_id, file in enumerate(file_list):
                    if file.endswith('.pkl') and not file.__contains__('data_step_before_remodelling') and not file.__contains__('features_per_time.pkl'):
                        if file.startswith('before_ablation'):
                            num_step = 0
                        else:
                            num_step = file.split('_')[2].split('.')[0]

                        # Load the state of the model
                        load_state(vModel, os.path.join(folder, c_dir, file))
                        vModel.set.VTK = True
                        vModel.set.OutputFolder = os.path.join(folder, c_dir)
                        vModel.geo.create_vtk_cell(vModel.set, vModel.numStep, 'Cells')
                        #vModel.geo.create_vtk_cell(vModel.set, vModel.numStep, 'Edges')
                        # Compute Surface Tension
                        kg_surface_area = KgSurfaceCellBasedAdhesion(vModel.geo)
                        kg_surface_area.compute_work(vModel.geo, vModel.set, None, False)

                        # Compute Volume
                        kg_volume = KgVolume(vModel.geo)
                        kg_volume.compute_work(vModel.geo, vModel.set, None, False)
                        vModel.geo.create_vtk_cell(vModel.set, vModel.numStep, 'Arrows_sa', -kg_surface_area.g)
                        vModel.geo.create_vtk_cell(vModel.set, vModel.numStep, 'Arrows_vol', -kg_volume.g)

                        # if image exists, skip
                        if os.path.exists(os.path.join(folder, c_dir, 'images', 'vModel_combined_' + str(num_step) + '.png')):
                            continue
                        output_folder = vModel.set.OutputFolder + '/images/'
                        try:
                            screenshot(vModel, output_folder)
                        except Exception as e:
                            print(f"Error taking screenshot for {file}: {e}")
                            continue

                # Create video
                create_video(os.path.join(folder, c_dir, 'images'), 'combined_')

else:
    specific_files = ['291', '302', '313', '324']
    c_dir = '/media/pablo/d7c61090-024c-469a-930c-f5ada47fb049/PabloVicenteMunuera/VertexModel/pyVertexModel/Result/07-15_155035_dWP1_scutoids_0.99_lVol_1.00e+00_kSubs_1.00e-01_lt_0.00e+00_refA0_9.20e-01_eARBarrier_8.00e-07_RemStiff_2_lS1_1.40e+00_lS2_7.00e-01_lS3_1.40e+00_ps_3.00e-05_lc_7.00e-05'
    specific_files = os.listdir(c_dir)
    # Filter specific files that end with '.pkl' and do not contain 'before_remodelling'
    specific_files = [f for f in specific_files if f.endswith('.pkl') and not 'before_remodelling' in f and not 'features_per_time.pkl' in f]
    # Sort specific files by date
    specific_files.sort(key=lambda x: os.path.getmtime(os.path.join(c_dir, x)))
    for specific_file in specific_files:
        print('------')
        print('Processing file: ' + specific_file)
        vModel = VertexModel(create_output_folder=False)
        load_state(vModel, c_dir + '/' + specific_file)
        vModel.set.VTK = True
        #vModel.geo.create_vtk_cell(vModel.set, vModel.numStep, 'Cells')
        vModel.geo.create_vtk_cell(vModel.set, vModel.numStep, 'Single_Cells')
        #vModel.geo.create_vtk_cell(vModel.set, vModel.numStep, 'Edges')

        # Compute Surface Tension
        kg_surface_area = KgSurfaceCellBasedAdhesion(vModel.geo)
        kg_surface_area.compute_work(vModel.geo, vModel.set, None, False)

        kg_volume = KgVolume(vModel.geo)
        kg_volume.compute_work(vModel.geo, vModel.set, None, False)

        vModel.geo.create_vtk_cell(vModel.set, vModel.numStep, 'Arrows_sa', -kg_surface_area.g)
        vModel.geo.create_vtk_cell(vModel.set, vModel.numStep, 'Arrows_vol', -kg_volume.g)

        global_id = int(np.floor(np.argmax(abs(kg_volume.g)) / 3))
        print(np.argmax(abs(kg_volume.g)) / 3)
        print(kg_volume.g[global_id*3:global_id*3+3])
        print(np.max(abs(kg_volume.g)))

        cell_id = -1
        for c_cell in vModel.geo.Cells:
            if np.isin(c_cell.globalIds, global_id).any():
                print('Cell: ' + str(c_cell.ID))
                cell_id = c_cell.ID
            if c_cell.AliveStatus is not None:
                for face in c_cell.Faces:
                    if np.isin(face.globalIds, global_id).any():
                        print('Cell: ' + str(c_cell.ID) + ', Face: ' + str(face.globalIds))

        if cell_id == -1:
            print('Cell not found for global_id: ' + str(global_id))
            continue

        # It is a scutoid face
        c_cell = vModel.geo.Cells[cell_id]
        tet = c_cell.T[c_cell.globalIds == global_id][0]
        if np.sum([True for t in tet if vModel.geo.Cells[t].AliveStatus is not None]) == 4:
            print('Scutoid face found in cell: ' + str(tet))
        else:
            print('Scutoid face NOT found in cell: ' + str(tet))

        for n_cell in tet:
            cc_cell = vModel.geo.Cells[n_cell]
            print('Cell: ' + str(cc_cell.ID))
            print('Volume: ' + str(cc_cell.Vol))
            print('Volume0: ' + str(cc_cell.Vol0))


        # output_folder = vModel.set.OutputFolder + '/images/'
        # screenshot(vModel, output_folder)