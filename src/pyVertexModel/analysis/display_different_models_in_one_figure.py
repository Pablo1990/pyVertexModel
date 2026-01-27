import numpy as np
import pyvista as pv
import matplotlib

from src.pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage import VertexModelVoronoiFromTimeImage
from src.pyVertexModel.geometry.geo import Geo
from src.pyVertexModel.util.utils import load_state

matplotlib.use('Agg')
import matplotlib.pyplot as plt

## Display different models in one figure

# files to load
folder = '/media/pablo/d7c61090-024c-469a-930c-f5ada47fb049/PabloVicenteMunuera/VertexModel/pyVertexModel/Input/images/'
output_folder = '/media/pablo/d7c61090-024c-469a-930c-f5ada47fb049/PabloVicenteMunuera/VertexModel/pyVertexModel/Result/'
files_to_load = ['dWP1_150cells_0.15.pkl', 'dWP1_150cells_1.5.pkl', 'dWP1_150cells_7.5.pkl', 'dWP1_150cells_15.0.pkl', 'dWP1_150cells_30.0.pkl']

geo = Geo()

# Create a figure
plotter = pv.Plotter(off_screen=True)

colormap_lim = [0.0001, 0.0006]

cells_to_hide = [1, 5, 14, 23, 35]

# Set an offset for each model to avoid overlap
offset = np.array([0.0, 0.0, 0.0])
for id_file, file in enumerate(files_to_load):
    load_state(geo, folder + file)
    geo.resize_tissue()
    for _, cell in enumerate(geo.Cells):
        if cell.AliveStatus == 1:
            # Load the VTK file as a pyvista mesh
            mesh = cell.create_pyvista_mesh(offset=offset)

            # Add the mesh to the plotter
            plotter.add_mesh(mesh, name=f'cell_{cell.ID}_{file}', scalars='Volume', lighting=True, cmap="pink",
                             clim=colormap_lim, show_edges=True, edge_color='white', edge_opacity=0.3)


    for _, cell in enumerate(geo.Cells):
        if cell.AliveStatus == 1:
            edge_mesh = cell.create_pyvista_edges(offset=offset)
            plotter.add_mesh(edge_mesh, name=f'edge_{cell.ID}_{file}', color='black', line_width=3,
                             render_lines_as_tubes=True)

#plotter.enable_parallel_projection()
    bounds = plotter.bounds
    x_size = bounds[1] - bounds[0]
    offset = np.array([x_size, 0.0, 0.0])

plotter.enable_image_style()
plotter.view_xz()

pos = plotter.camera.position
focal = plotter.camera.focal_point

# Move camera to the side and above
plotter.camera.position = (pos[0] + 2.0, pos[1], pos[2] + 1.0)

# Optionally, adjust focal point if needed
plotter.camera.focal_point = (focal[0], focal[1], focal[2])
plotter.camera.zoom(1.3)

# Take a screenshot with a higher resolution
img = plotter.screenshot(scale=10)
plotter.close()

# Create figure
fig = plt.figure(figsize=(10, 10), dpi=2000)
plt.imshow(img)
plt.axis('off')
plt.savefig(output_folder + 'dWP1_all_tissues.png', bbox_inches='tight')
plt.close(fig)