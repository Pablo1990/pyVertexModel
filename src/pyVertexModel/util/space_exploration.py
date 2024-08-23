import optuna
from src.pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage import VertexModelVoronoiFromTimeImage

def objective(trial):
    # Define the parameter space
    params = {
        'nu': trial.suggest_uniform('nu', 0.01, 1),
        'lambdaV': trial.suggest_uniform('lambdaV', 0.01, 100),
        'ref_V0': trial.suggest_uniform('ref_V0', 0.5, 2),
        'kSubstrate': trial.suggest_uniform('kSubstrate', 0.01, 100),
        'cLineTension': trial.suggest_uniform('cLineTension', 1e-6, 1e-2),
        'cLineTension_external': trial.suggest_uniform('cLineTension_external', 1e-6, 1e-2),
        'ref_A0': trial.suggest_uniform('ref_A0', 0.5, 2),
        'lambdaS1': trial.suggest_uniform('lambdaS1', 0.01, 100),
        'lambdaS2': trial.suggest_uniform('lambdaS2', 0.01, 100),
        'lambdaS3': trial.suggest_uniform('lambdaS3', 0.01, 100),
    }

    # Initialize the model with the parameters
    vModel = VertexModelVoronoiFromTimeImage()

    # Set the parameters
    vModel.set.nu = params['nu']
    vModel.set.lambdaV = params['lambdaV']
    vModel.set.ref_V0 = params['ref_V0']
    vModel.set.kSubstrate = params['kSubstrate']
    vModel.set.cLineTension = params['cLineTension']
    vModel.set.cLineTension_external = params['cLineTension_external']
    vModel.set.ref_A0 = params['ref_A0']
    vModel.set.lambdaS1 = params['lambdaS1']
    vModel.set.lambdaS2 = params['lambdaS2']
    vModel.set.lambdaS3 = params['lambdaS3']

    # Run the simulation
    vModel.iterate_over_time()

    # Return a metric to minimize
    error = vModel.calculate_error()
    return error

# Create a study object and optimize the objective function
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=200, n_jobs=5)

print("Best parameters:", study.best_params)