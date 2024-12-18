import itertools
import os
from concurrent.futures import ThreadPoolExecutor

from src import PROJECT_DIRECTORY
from src.pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage import VertexModelVoronoiFromTimeImage
from src.pyVertexModel.analysis.analyse_simulation import analyse_simulation
from src.pyVertexModel.util.utils import load_state

variables_to_change = ['kSubstrate', 'Remodelling', 'purseStringStrength', 'lateralCablesStrength']

# Create all possible combinations of variables
combinations_of_variables = []
for i in range(1, len(variables_to_change) + 1):
    combinations_of_variables.extend(itertools.combinations(variables_to_change, i))

def run_simulation(combination):
    vModel = VertexModelVoronoiFromTimeImage()
    vModel.initialize()

    vModel = VertexModelVoronoiFromTimeImage()
    load_state(vModel,
               os.path.join(PROJECT_DIRECTORY, 'Result/'
                                               'new_reference/'
                                               'before_ablation.pkl'))

    vModel.set.wing_disc()
    vModel.set.wound_default()
    vModel.set.OutputFolder = os.path.join(PROJECT_DIRECTORY, 'Result/final_results_2/60_mins_no_{}'.format('_no_'.join(combination)))
    for variable in combination:
        if variable == 'Remodelling':
            setattr(vModel.set, variable, False)
        else:
            setattr(vModel.set, variable, 0)
    vModel.set.dt0 = None
    vModel.set.dt = None
    vModel.set.update_derived_parameters()
    vModel.iterate_over_time()
    analyse_simulation(vModel.set.OutputFolder)

run_simulation(combinations_of_variables[1])

# Run simulations in parallel
with ThreadPoolExecutor(max_workers=2) as executor:
    executor.map(run_simulation, combinations_of_variables)