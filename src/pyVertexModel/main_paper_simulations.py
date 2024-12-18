import itertools
import os
from concurrent.futures import ThreadPoolExecutor

from src import PROJECT_DIRECTORY
from src.pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage import VertexModelVoronoiFromTimeImage
from src.pyVertexModel.analysis.analyse_simulation import analyse_simulation
from src.pyVertexModel.util.utils import load_state

def run_simulation(combination):
    vModel = VertexModelVoronoiFromTimeImage(create_output_folder=False)
    load_state(vModel,
               os.path.join(PROJECT_DIRECTORY, 'Result/'
                                               'new_reference/'
                                               'before_ablation.pkl'))

    vModel.set.wing_disc()
    vModel.set.wound_default()
    # Based on Tissue Fluidity paper from Rob Tetley et al. 2019
    if combination == 'Mbs':
        #“In Mbs RNAi wing discs, vertex recoil rates are increased by 40% in the tissue and decreased by 21% in the purse string compared to WT”
        vModel.set.lambdaS1 = vModel.set.lambdaS1 + (vModel.set.lambdaS1 * 0.40)
        vModel.set.purseStringStrength = vModel.set.purseStringStrength - (vModel.set.purseStringStrength * 0.21)
        vModel.set.lateralCablesStrength = vModel.set.lateralCablesStrength - (vModel.set.lateralCablesStrength * 0.21)
        vModel.set.OutputFolder = os.path.join(PROJECT_DIRECTORY, 'Result/final_results_2/60_mins_Mbs')
    elif combination == 'Rok':
        #“In Rok RNAi wing discs, vertex recoil rates are reduced by 51% in the tissue and 35% in the purse string compared to WT”
        vModel.set.lambdaS1 = vModel.set.lambdaS1 - (vModel.set.lambdaS1 * 0.51)
        vModel.set.purseStringStrength = vModel.set.purseStringStrength - (vModel.set.purseStringStrength * 0.35)
        vModel.set.lateralCablesStrength = vModel.set.lateralCablesStrength - (vModel.set.lateralCablesStrength * 0.35)
        vModel.set.OutputFolder = os.path.join(PROJECT_DIRECTORY, 'Result/final_results_2/60_mins_Rok')
    else:
        vModel.set.OutputFolder = os.path.join(PROJECT_DIRECTORY, 'Result/final_results_2/60_mins_no_{}'.format('_no_'.join(combination)))
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
    vModel.iterate_over_time()
    analyse_simulation(vModel.set.OutputFolder)

variables_to_change = ['kSubstrate', 'Remodelling', 'purseStringStrength', 'lateralCablesStrength']

# Create all possible combinations of variables
combinations_of_variables = []
for i in range(1, len(variables_to_change) + 1):
    combinations_of_variables.extend(itertools.combinations(variables_to_change, i))

combinations_of_variables.extend(['Mbs'])
combinations_of_variables.extend(['Rok'])

run_simulation(combinations_of_variables[-1])

# Run simulations in parallel
with ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(run_simulation, combinations_of_variables)