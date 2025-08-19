# Get stats from the space exploration study
import os

import optuna
import plotly


output_directory = '/media/pablo/d7c61090-024c-469a-930c-f5ada47fb049/PabloVicenteMunuera/VertexModel/pyVertexModel/Result/'
error_type = '_gr_'
if error_type is not None:
    study_name = "VertexModel" + error_type  # Unique identifier of the study.
else:
    study_name = "VertexModel"
storage_name = "sqlite:///{}.db".format("VertexModel")

study = optuna.create_study(study_name=study_name, storage=storage_name, direction='minimize',
                            load_if_exists=True)

output_dir_study = os.path.join(output_directory, study_name)

# Create a dataframe from the study.
df = study.trials_dataframe()
df.to_excel(output_dir_study + '/df.xlsx')

# Plot the contour of the study
fig = optuna.visualization.plot_contour(study, params=["x", "y"])
plotly.io.write_image(fig, output_dir_study + 'countour.png')

# Plot the parallel coordinates of the study
fig = optuna.visualization.plot_parallel_coordinate(study)
plotly.io.write_image(fig, output_dir_study + 'parallel_coordinate.png')