# Get stats from the space exploration study
import optuna

from pyVertexModel.util.space_exploration import plot_optuna_all

output_directory = '/media/pablo/d7c61090-024c-469a-930c-f5ada47fb049/PabloVicenteMunuera/VertexModel/pyVertexModel/Result/'
#error_type = '_gr_all_parameters_'
#error_type = '_K_InitialRecoil_fixed_LT_var_A0_S1_'
#error_type = '_K_InitialRecoil_NoLT_var_A0_S1_'
#error_type = '_K_InitialRecoil_NoLT_var_A0_S1_S2_S3_'
error_type = '_K_InitialRecoil_allParams_smallRange_'
if error_type is not None:
    study_name = "VertexModel" + error_type  # Unique identifier of the study.
else:
    study_name = "VertexModel"

storage_name = "sqlite:////media/pablo/d7c61090-024c-469a-930c-f5ada47fb049/PabloVicenteMunuera/VertexModel/pyVertexModel/src/pyVertexModel/VertexModel.db"

study = optuna.create_study(study_name=study_name, storage=storage_name, direction='minimize',
                            load_if_exists=True)

plot_optuna_all(output_directory, study_name, study)
