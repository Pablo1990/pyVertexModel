import itertools
import os
import sys

from src import PROJECT_DIRECTORY
from src.pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage import VertexModelVoronoiFromTimeImage
from src.pyVertexModel.analysis.analyse_simulation import analyse_simulation
from src.pyVertexModel.util.utils import load_state

def run_simulation(combination, output_results_dir='Result/'):
    """
    Run simulation with the given combination of variables.
    :param output_results_dir: 
    :param combination:
    :return:
    """  # output directory

    vModel = VertexModelVoronoiFromTimeImage(create_output_folder=False)
    if combination == 'WT' or combination == 'Mbs' or combination == 'Rok' or combination == 'Talin' or combination == 'IntegrinDN':
        output_folder = os.path.join(PROJECT_DIRECTORY, output_results_dir, '60_mins_{}'.format(combination))
    else:
        output_folder = os.path.join(PROJECT_DIRECTORY, output_results_dir, '60_mins_no_{}'.format('_no_'.join(combination)))

    # Check if output_folder exists
    if not os.path.exists(output_folder):
        load_state(vModel,
                   os.path.join(PROJECT_DIRECTORY, output_results_dir,
                                                   'before_ablation.pkl'))

        vModel.set.wing_disc()
        vModel.set.wound_default()
        vModel.set.nu_bottom = vModel.set.nu * 600
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
        elif combination == 'Talin':
            # Control: 0.66; Talin: 0.81
            vModel.set.kSubstrate = vModel.set.kSubstrate * 0
            vModel.set.lateralCablesStrength = vModel.set.lateralCablesStrength * 0.81/0.66
        elif combination == 'IntegrinDN':
            # Control: 0.66 ; IntegrinDN: 0.56
            vModel.set.kSubstrate = vModel.set.kSubstrate * 0
            vModel.set.lateralCablesStrength = vModel.set.lateralCablesStrength * 0.56/0.66
        else:
            for variable in combination:
                if variable == 'Remodelling':
                    vModel.set.Remodelling = False
                    vModel.set.RemodelStiffness = 2
                else:
                    setattr(vModel.set, variable, 0)
        vModel.set.dt0 = None
        vModel.set.dt = None
        vModel.set.update_derived_parameters()
        vModel.set.redirect_output()
    else:
        print("Output folder already exists: {}".format(output_folder))
        # Load last modified pkl file
        name_last_pkl_file = sorted(
            [f for f in os.listdir(output_folder) if f.endswith('.pkl') and not 'before_remodelling' in f
             and f.startswith('data_step')],
            key=lambda x: os.path.getmtime(os.path.join(output_folder, x))
        )[-1]
        load_state(vModel, os.path.join(output_folder, name_last_pkl_file))
        vModel.set.OutputFolder = output_folder
        vModel.set.redirect_output()
        if vModel.t > vModel.set.tend:
            analyse_simulation(vModel.set.OutputFolder)
            return

    vModel.iterate_over_time()
    analyse_simulation(vModel.set.OutputFolder)

variables_to_change = ['kSubstrate', 'Remodelling', 'purseStringStrength', 'lateralCablesStrength']

combinations_of_variables = []
for i in range(1, len(variables_to_change) + 1):
    combinations_of_variables.extend(itertools.combinations(variables_to_change, i))

combinations_of_variables.insert(0, 'Mbs')
combinations_of_variables.insert(0, 'Rok')
combinations_of_variables.insert(0, 'Talin')
combinations_of_variables.insert(0, 'IntegrinDN')
combinations_of_variables.insert(0, 'WT')

if __name__ == '__main__':
    index = int(sys.argv[1])
    # Check if there are two arguments
    if len(sys.argv) == 2:
        run_simulation(combinations_of_variables[index])
    else:
        run_simulation(combinations_of_variables[index], sys.argv[2])