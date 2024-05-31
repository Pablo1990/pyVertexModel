import os
import imageio
import pyvista as pv

from src.pyVertexModel.algorithm.vertexModel import VertexModel
from src.pyVertexModel.util.utils import load_state

folder = '/media/pablo/d7c61090-024c-469a-930c-f5ada47fb049/PabloVicenteMunuera/VertexModel/Results/Relevant/'
folder_2 = '/media/pablo/d7c61090-024c-469a-930c-f5ada47fb049/PabloVicenteMunuera/VertexModel/pyVertexModel/Result/'
files = '05-30_132736_VertexModelTime_Cells_150_visc_100_lVol_0.01_kSubs_1_lt_0.0025_noise_0_brownian_0_eTriAreaBarrier_0_eARBarrier_0.001_RemStiff_0.98_lS1_0.01_lS2_0.01_lS3_0.01_pString_2.5_tol_5.0'
vtk_dir = folder + files
vtk_dir_output = folder_2 + files

max_t = 0
for file_id, file in enumerate(os.listdir(vtk_dir)):
    if file.endswith('.pkl') and not file.__contains__('data_step_before_remodelling'):
        vModel = VertexModel(create_output_folder=False)

        # Load the state of the model
        load_state(vModel, os.path.join(vtk_dir, file))

        # if directory called 'cells' does not exist, create it
        vModel.set.VTK = True
        vModel.geo.create_vtk_cell(vModel.set, vModel.numStep, 'Cells')

# Directory where your VTK files are stored
vtk_dir = os.path.join(vtk_dir_output, 'Cells')

# Create a temporary directory to store the images
temp_dir = '/media/pablo/d7c61090-024c-469a-930c-f5ada47fb049/PabloVicenteMunuera/VertexModel/pyVertexModel/tmp'

images = []

for i in range(max_t):
    # Get a list of VTK files
    vtk_files = [f for f in os.listdir(vtk_dir) if f.endswith(f'{i:04d}.vtk') and not f.startswith('Cells.0001')
                 and not f.startswith('Cells.0000') and not f.startswith('Cells.0002') and not f.startswith(
        'Cells.0003')
                 and not f.startswith('Cells.0004') and not f.startswith('Cells.0005') and not f.startswith(
        'Cells.0006')
                 and not f.startswith('Cells.0007') and not f.startswith('Cells.0008') and not f.startswith(
        'Cells.0009')]

    vtk_files.sort()  # Ensure files are sorted in order
    if len(vtk_files) == 0:
        continue

    # Create a plotter
    plotter = pv.Plotter(off_screen=True)

    for _, file in enumerate(vtk_files):
        # Load the VTK file as a pyvista mesh
        mesh = pv.read(os.path.join(vtk_dir, file))

        # Add the mesh to the plotter
        plotter.add_mesh(mesh, scalars='ID', show_edges=True, edge_color='black',
                         lighting=True, cmap='gist_ncar')

    # Render the scene and capture a screenshot
    img = plotter.screenshot()

    # Save the image to a temporary file
    temp_file = os.path.join(temp_dir, f'temp_{i}.png')
    imageio.imwrite(temp_file, img)

    # Add the temporary file to the list of images
    images.append(imageio.v2.imread(temp_file))

# Create a movie from the images
imageio.mimwrite(folder + files + 'movie.avi', images, fps=30)

# Clean up the temporary files
for file in os.listdir(temp_dir):
    os.remove(os.path.join(temp_dir, file))
os.rmdir(temp_dir)

# Remove 'Cells' directory
for file in os.listdir(vtk_dir):
    os.remove(os.path.join(vtk_dir, file))
os.rmdir(vtk_dir)
