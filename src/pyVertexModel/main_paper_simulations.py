import itertools
import logging
import os
import sys

import numpy as np

from src import PROJECT_DIRECTORY
from src.pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage import VertexModelVoronoiFromTimeImage
from src.pyVertexModel.analysis.analyse_simulation import analyse_simulation
from src.pyVertexModel.util.utils import load_state

logger = logging.getLogger("pyVertexModel")

def run_simulation(combination, output_results_dir='Result/', length="60_mins"):
    """
    Run simulation with the given combination of variables.
    :param length: 
    :param output_results_dir: 
    :param combination:
    :return:
    """
    # output directory
    vModel = VertexModelVoronoiFromTimeImage(create_output_folder=False, set_option='wing_disc_equilibrium')
    if (combination == 'WT' or combination == 'Mbs' or combination == 'Rok' or
            combination == 'Talin' or combination == 'IntegrinDN' or
            combination == 'ShibireTS' or combination == 'WT_substrate_gone_40_mins' or
            combination == 'Talin_with_substrate' or combination == 'IntegrinDN_with_substrate'):
        output_folder = os.path.join(PROJECT_DIRECTORY, output_results_dir, length + '_{}'.format(combination))
    else:
        output_folder = os.path.join(PROJECT_DIRECTORY, output_results_dir, length + '_no_{}'.format('_no_'.join(combination)))

    # Check if output_folder exists
    if not os.path.exists(output_folder):
        load_state(vModel,
                   os.path.join(PROJECT_DIRECTORY, output_results_dir,
                                                   'before_ablation.pkl'))

        #vModel.set.wing_disc()
        #vModel.set.wound_default()
        #cells_to_ablate = vModel.set.cellsToAblate
        if getattr(vModel.set, 'model_name', None) is None:
            vModel.set.model_name = 'in_silico_movie'
        elif vModel.set.model_name == 'wing_disc_real_bottom_left':
            cells_to_ablate = np.array([0, 1, 2, 3, 4, 7, 8, 10, 13])
        elif vModel.set.model_name == 'wing_disc_real_top_right':
            cells_to_ablate = np.array([0, 1, 2, 3, 4, 5, 6, 7, 10, 13, 15])
        elif vModel.set.model_name == 'wing_disc_real':
            cells_to_ablate = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        elif vModel.set.model_name.startswith('dWL6'):
            cells_to_ablate = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13])
        else:
            cells_to_ablate = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        vModel.set.cellsToAblate = cells_to_ablate
        vModel.geo.cellsToAblate = cells_to_ablate

        # vModel.set.EnergyBarrierA = True
        # vModel.set.Beta = 1
        # vModel.geo.update_barrier_tri0(factor=10, count_lateral_faces=False)
        # print(f"Barrier tri0: {vModel.geo.BarrierTri0}")
        # vModel.set.lambdaB = vModel.geo.BarrierTri0 * 10

        # Recompute reference lengths to compare lateral cables and purse string effectively
        vModel.geo.compute_edge_length_0(default_value=1.0)

        # Additional viscosity for the bottom vertices based on the cell height
        vModel.set.nu_bottom = vModel.set.nu + (vModel.set.nu * (600 * (vModel.set.CellHeight / 15) ** 2))

        vModel.set.OutputFolder = output_folder
        if combination == 'WT':
            pass
        elif combination == 'Mbs':
            vModel.set.lambdaS1 = vModel.set.lambdaS1 + (vModel.set.lambdaS1 * 0.40)
            vModel.set.purseStringStrength = vModel.set.purseStringStrength - (vModel.set.purseStringStrength * 0.21)
            vModel.set.lateralCablesStrength = vModel.set.lateralCablesStrength - (vModel.set.lateralCablesStrength * 0.21)
        elif combination == 'Rok':
            vModel.set.lambdaS1 = vModel.set.lambdaS1 - (vModel.set.lambdaS1 * 0.51)
            vModel.set.purseStringStrength = vModel.set.purseStringStrength - (vModel.set.purseStringStrength * 0.35)
            vModel.set.lateralCablesStrength = vModel.set.lateralCablesStrength - (vModel.set.lateralCablesStrength * 0.35)
        elif combination == 'ShibireTS':
            vModel.set.purseStringStrength = vModel.set.purseStringStrength - (vModel.set.purseStringStrength * 0.4)
        elif combination == 'Talin':
            vModel.set.kSubstrate = vModel.set.kSubstrate * 0
            vModel.set.lateralCablesStrength = vModel.set.lateralCablesStrength * 4.3/3.1
        elif combination == 'Talin_with_substrate':
            vModel.set.lateralCablesStrength = vModel.set.lateralCablesStrength * 4.3/3.1
        elif combination == 'IntegrinDN':
            vModel.set.kSubstrate = vModel.set.kSubstrate * 0
            vModel.set.lateralCablesStrength = vModel.set.lateralCablesStrength * 2.1/3.1
        elif combination == 'IntegrinDN_with_substrate':
            vModel.set.lateralCablesStrength = vModel.set.lateralCablesStrength * 2.1/3.1
        else:
            for variable in combination:
                if variable == 'Remodelling':
                    vModel.set.Remodelling = False
                    vModel.set.RemodelStiffness = 2
                else:
                    setattr(vModel.set, variable, 0)

        if length == '120_mins':
            vModel.set.tend = 120
        vModel.set.dt0 = None
        vModel.set.dt = None
        vModel.set.update_derived_parameters()
        vModel.set.redirect_output()
    else:
        print("Output folder already exists: {}".format(output_folder))
        # if os.path.exists(os.path.join(output_folder, 'features_per_time.pkl')):
        #     print("Analysis already done for this folder: {}".format(output_folder))
        #     return
        # Load last modified pkl file
        name_last_pkl_file = sorted(
            [f for f in os.listdir(output_folder) if f.endswith('.pkl') and not 'before_remodelling' in f
             and f.startswith('data_step')],
            key=lambda x: os.path.getmtime(os.path.join(output_folder, x))
        )[-1]
        load_state(vModel, os.path.join(output_folder, name_last_pkl_file))
        vModel.set.OutputFolder = output_folder
        vModel.set.redirect_output()

        if combination == 'WT_substrate_gone_40_mins':
            vModel.set.kSubstrate = 0

        if length == '120_mins':
            vModel.set.tend = 120

        if vModel.t > vModel.set.tend:
            print("Performing analysis for folder...")
            analyse_simulation(vModel.set.OutputFolder)
            return

    #try:
    vModel.geo.update_measures()
    vModel.iterate_over_time()
    analyse_simulation(vModel.set.OutputFolder)
    #except Exception as e:
    #    logger.info(f"Error in simulation: {e}")


variables_to_change = ['kSubstrate', 'Remodelling', 'purseStringStrength', 'lateralCablesStrength']

combinations_of_variables = []
for i in range(1, len(variables_to_change) + 1):
    combinations_of_variables.extend(itertools.combinations(variables_to_change, i))

#combinations_of_variables.insert(0, 'IntegrinDN_with_substrate')
#combinations_of_variables.insert(0, 'Talin_with_substrate')
#combinations_of_variables.insert(0, 'ShibireTS')
#combinations_of_variables.insert(0, 'Mbs')
#combinations_of_variables.insert(0, 'Rok')
#combinations_of_variables.insert(0, 'Talin')
#combinations_of_variables.insert(0, 'IntegrinDN')
combinations_of_variables.insert(0, 'WT')

if __name__ == '__main__':
    index = int(sys.argv[1])
    # Check if there are two arguments
    if len(sys.argv) == 2:
        run_simulation(combinations_of_variables[index])
    elif len(sys.argv) == 4:
        run_simulation(combinations_of_variables[index], sys.argv[2], sys.argv[3])
    else:
        run_simulation(combinations_of_variables[index], sys.argv[2])
