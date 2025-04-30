import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy import stats

from src import PROJECT_DIRECTORY
from src.pyVertexModel.Kg.kgContractility import KgContractility
from src.pyVertexModel.Kg.kgSurfaceCellBasedAdhesion import KgSurfaceCellBasedAdhesion
from src.pyVertexModel.Kg.kgTriAREnergyBarrier import KgTriAREnergyBarrier
from src.pyVertexModel.Kg.kgVolume import KgVolume
from src.pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage import VertexModelVoronoiFromTimeImage
from src.pyVertexModel.analysis.analyse_simulation import analyse_simulation
from src.pyVertexModel.geometry.cell import compute_2d_circularity
from src.pyVertexModel.util.utils import load_state, save_state, save_variables, load_variables


def compute_aspect_ratios(cell):
    """Compute 2D and 3D aspect ratios for a cell"""
    aspect_2d_top = cell.compute_2d_aspect_ratio(filter_location=0)
    aspect_2d_bottom = cell.compute_2d_aspect_ratio(filter_location=2)

    if cell.axes_lengths is None:
        cell.compute_principal_axis_length()
    aspect_3d = max(cell.axes_lengths) / min(cell.axes_lengths)

    return {
        'aspect_2d_top': aspect_2d_top,
        'aspect_2d_bottom': aspect_2d_bottom,
        'aspect_3d': aspect_3d
    }


def check_inverted(cell):
    """Check for inverted cells using signed volume"""
    return cell.compute_volume() < 0


def compute_min_angles(cell):
    """Compute minimum angles in all triangular faces"""
    min_angles = []
    for face in cell.Faces:
        for tri in face.Tris:
            v0 = cell.Y[tri.Edge[0]]
            v1 = cell.Y[tri.Edge[1]]
            v2 = face.Centre

            vec1 = v1 - v0
            vec2 = v2 - v0
            vec3 = v2 - v1

            vec1 = vec1 / np.linalg.norm(vec1)
            vec2 = vec2 / np.linalg.norm(vec2)
            vec3 = vec3 / np.linalg.norm(vec3)

            angle1 = np.arccos(np.clip(np.dot(vec1, vec2), -1.0, 1.0))
            angle2 = np.arccos(np.clip(np.dot(-vec1, vec3), -1.0, 1.0))
            angle3 = np.pi - angle1 - angle2

            min_angle = min(angle1, angle2, angle3)
            min_angles.append(np.degrees(min_angle))

    return min(min_angles) if min_angles else 0


def export_diagnostic_vtk(Geo, output_path, time):
    """Export VTK with geometry diagnostic fields"""
    try:
        import pyvista as pv
    except ImportError:
        print("Warning: PyVista not available - skipping VTK export")
        return False

    try:
        # Initialize lists to store cell data
        aspect_3d_values = []
        aspect_2d_top_values = []
        aspect_2d_bottom_values = []
        is_inverted_values = []
        min_angle_values = []
        cell_id_values = []

        # First pass: collect all cell data
        for cell in Geo.Cells:
            if cell.AliveStatus is None:
                continue

            aspect_ratios = compute_aspect_ratios(cell)
            is_inverted = check_inverted(cell)
            min_angle = compute_min_angles(cell)

            # Get the number of triangles in this cell
            num_tris = sum(len(face.Tris) for face in cell.Faces)

            # Repeat values for each triangle in the cell
            aspect_3d_values.extend([aspect_ratios['aspect_3d']] * num_tris)
            aspect_2d_top_values.extend([aspect_ratios['aspect_2d_top']] * num_tris)
            aspect_2d_bottom_values.extend([aspect_ratios['aspect_2d_bottom']] * num_tris)
            is_inverted_values.extend([float(is_inverted)] * num_tris)
            min_angle_values.extend([min_angle] * num_tris)
            cell_id_values.extend([cell.ID] * num_tris)

        # Create the mesh (assuming create_pyvista_mesh() creates the full mesh)
        dataset = Geo.Cells[0].create_pyvista_mesh()  # Start with first cell
        for cell in Geo.Cells[1:]:
            if cell.AliveStatus is not None:
                dataset += cell.create_pyvista_mesh()

        # Add cell data (must match number of cells)
        dataset.cell_data['aspect_3d'] = aspect_3d_values
        dataset.cell_data['aspect_2d_top'] = aspect_2d_top_values
        dataset.cell_data['aspect_2d_bottom'] = aspect_2d_bottom_values
        dataset.cell_data['is_inverted'] = is_inverted_values
        dataset.cell_data['min_angle'] = min_angle_values
        dataset.cell_data['cell_id'] = cell_id_values

        dataset.save(output_path)
        return True
    except Exception as e:
        print(f"Error exporting VTK: {str(e)}")
        return False
# Add threshold constants
ASPECT_RATIO_3D_THRESHOLD = 5.0
ASPECT_RATIO_2D_THRESHOLD = 3.0
MIN_ANGLE_THRESHOLD = 5.0
ENERGY_THRESHOLD_FACTOR = 2.0


def track_cell_metrics(vModel, Geo, kg_lt, kg_surface_area, kg_volume, kg_tri_ar, time):
    """Track comprehensive metrics for all cells at each time step"""
    cell_data = []

    for cell_id, cell in enumerate(Geo.Cells):
        if cell.AliveStatus is None:
            continue

        # Energy components
        energies = {
            'time': time,
            'cell_id': cell_id,
            'energy_lt': kg_lt.energy_per_cell[cell_id],
            'energy_surface': kg_surface_area.energy_per_cell[cell_id],
            'energy_volume': kg_volume.energy_per_cell[cell_id],
            'energy_tri_ar': kg_tri_ar.energy_per_cell[cell_id],
            'total_energy': (kg_lt.energy_per_cell[cell_id] +
                             kg_surface_area.energy_per_cell[cell_id] +
                             kg_volume.energy_per_cell[cell_id] +
                             kg_tri_ar.energy_per_cell[cell_id])
        }

        # Geometry metrics
        aspect_ratios = compute_aspect_ratios(cell)
        geometry = {
            'aspect_3d': aspect_ratios['aspect_3d'],
            'aspect_2d_top': aspect_ratios['aspect_2d_top'],
            'aspect_2d_bottom': aspect_ratios['aspect_2d_bottom'],
            'min_angle': compute_min_angles(cell),
            'is_inverted': check_inverted(cell),
            'Area': cell.compute_area(),
            'Area_top': cell.compute_area(location_filter=0),
            'Area_bottom': cell.compute_area(location_filter=2),
            'Area_cellcell': cell.compute_area(location_filter=1),
            'Volume': cell.compute_volume(),
            'Height': cell.compute_height(),
            'Width': cell.compute_width(),
            'Length': cell.compute_length(),
            'Perimeter': cell.compute_perimeter(),
            '2D_circularity_top': compute_2d_circularity(cell.compute_area(location_filter=0),
                                                         cell.compute_perimeter(filter_location=0)),
            '2d_circularity_bottom': compute_2d_circularity(cell.compute_area(location_filter=2),
                                                            cell.compute_perimeter(filter_location=2)),
            '2D_aspect_ratio_top': cell.compute_2d_aspect_ratio(filter_location=0),
            '2D_aspect_ratio_bottom': cell.compute_2d_aspect_ratio(filter_location=2),
            '2D_aspect_ratio_cellcell': cell.compute_2d_aspect_ratio(filter_location=1),
            'Sphericity': cell.compute_sphericity(),
            'Elongation': cell.compute_elongation(),
            'Ellipticity': cell.compute_ellipticity(),
            'Tilting': cell.compute_tilting(),
            'Perimeter_top': cell.compute_perimeter(filter_location=0),
            'Perimeter_bottom': cell.compute_perimeter(filter_location=2),
            'Perimeter_cellcell': cell.compute_perimeter(filter_location=1),
        }

        # Topology metrics
        topology = {
            'neighbours_3d': len(cell.compute_neighbours()),
            'neighbours_apical': len(cell.compute_neighbours(location_filter='Top')),
            'neighbours_basal': len(cell.compute_neighbours(location_filter='Bottom')),
            'Scutoid': int(cell.is_scutoid()),
        }

        cell_data.append({**energies, **geometry, **topology})

    return pd.DataFrame(cell_data)


vModel = VertexModelVoronoiFromTimeImage(create_output_folder=False)
# Change this folder accordingly (it doesn't really matter the name of the folder, so feel free to rename it
output_folder = os.path.join(PROJECT_DIRECTORY, 'Result/04-29_121205_wing_disc_real_noise_0.00e+00_lVol_1.00e+01_kSubs_1.00e-01_lt_0.00e+00_refA0_9.20e-01_eARBarrier_8.00e-07_RemStiff_0.7_lS1_1.40e+00_lS2_1.40e-01_lS3_1.40e+00_ps_3.00e-05_lc_7.00e-05/')

# Load
if os.path.exists(os.path.join(output_folder, 'megha_analysis.pkl')):
    print('Loading previous analysis')
    data = load_variables(os.path.join(output_folder, 'megha_analysis.pkl'))
    geometry_issues_all_times = data['geometry_issues_all_times']
    energy_lt_cells = data['energy_lt_cells']
    energy_surface_cells = data['energy_surface_cells']
    energy_volume_cells = data['energy_volume_cells']
    energy_tri_ar_cells = data['energy_tri_ar_cells']
    neighbours_3d_cells = data['neighbours_3d_cells']
    neighbours_apical_cells = data['neighbours_apical_cells']
    neighbours_basal_cells = data['neighbours_basal_cells']
    num_intercalations = data['num_intercalations']
    apical_flip_times = data['apical_flip_times']
    basal_flip_times = data['basal_flip_times']
    scutoid_cells = data['scutoid_cells']
    all_cell_metrics = data['all_cell_metrics']
    times = data['times']
    time = data['times'][-1]
    if os.path.exists(os.path.join(output_folder, 'before_ablation.pkl')):
        print('Loading previous state')
        load_state(vModel, os.path.join(output_folder, 'before_ablation.pkl'))
    else:
        # Find the last file
        last_file = sorted(
            [f for f in os.listdir(output_folder) if f.endswith('.pkl') and not 'before_remodelling' in f and f.startswith('data_step_')],
            key=lambda x: os.path.getmtime(os.path.join(output_folder, x))
        )[-1]
        load_state(vModel, os.path.join(output_folder, last_file))
    Geo = vModel.geo
    Set = vModel.set
else:
    # Sort files by date
    all_files = os.listdir(output_folder)
    all_files = sorted(all_files, key=lambda x: os.path.getmtime(os.path.join(output_folder, x)))

    # Time here means just the number of intercalations or file number
    time = 0

    times = []

    # Energy for a specific cell. You can change the cell_id to any other cell you want to analyse until 149
    geometry_issues_all_times = []
    diagnostic_vtk_files = []
    energy_lt_cells = []
    energy_surface_cells = []
    energy_volume_cells = []
    energy_tri_ar_cells = []
    neighbours_3d_cells = []
    neighbours_apical_cells = []
    neighbours_basal_cells = []
    num_intercalations = []
    apical_flip_times = []
    basal_flip_times = []
    scutoid_cells = []
    all_cell_metrics = []

    for file_id, file in enumerate(all_files):
        if file.endswith('.pkl') and not file.__contains__(
                'data_step_before_remodelling') and not file.__contains__('recoil') and file.startswith('data_step_'):
            # Load the state of the model
            load_state(vModel, os.path.join(output_folder, file))

             #get Geo from the loaded model
            Geo = vModel.geo
            Set = vModel.set
            Set.currentT = 0
            # Right after load_state() call
            geometry_issues = {
                'time': time,
                'distorted_cells': [],
                'inverted_cells': [],
                'small_angle_cells': [],
                'high_energy_cells': []
            }

            # Cell-by-cell geometry checks
            for cell_id, cell in enumerate(Geo.Cells):
                if cell.AliveStatus is None:
                    continue

                # Aspect ratio checks
                aspect_ratios = compute_aspect_ratios(cell)
                if (aspect_ratios['aspect_3d'] > ASPECT_RATIO_3D_THRESHOLD or
                        aspect_ratios['aspect_2d_top'] > ASPECT_RATIO_2D_THRESHOLD or
                        aspect_ratios['aspect_2d_bottom'] > ASPECT_RATIO_2D_THRESHOLD):
                    geometry_issues['distorted_cells'].append({
                        'cell_id': cell_id,
                        **aspect_ratios
                    })

                # Inversion check
                if check_inverted(cell):
                    geometry_issues['inverted_cells'].append(cell_id)

                # Small angle check
                min_angle = compute_min_angles(cell)
                if min_angle < MIN_ANGLE_THRESHOLD:
                    geometry_issues['small_angle_cells'].append({
                        'cell_id': cell_id,
                        'min_angle': min_angle
                    })

            geometry_issues_all_times.append(geometry_issues)

            # Track intercalations
            if hasattr(vModel.geo, 'num_flips_this_step'):
                num_intercalations.append(vModel.geo.num_flips_this_step)
            else:
                num_intercalations.append(0)  # Default if not tracking flips

            # Track apical vs basal flips
            if hasattr(vModel.geo, 'apical_flips_this_step'):
                apical_flip_times.append(vModel.geo.apical_flips_this_step)
                basal_flip_times.append(vModel.geo.basal_flips_this_step)
            else:
                apical_flip_times.append(0)
                basal_flip_times.append(0)

            print('File: ', file)

            # Export images
            # vModel.set.export_images = True
            # temp_dir = os.path.join(vModel.set.OutputFolder, 'images')
            # screenshot(vModel, temp_dir)

            # Analyse the simulation
            #all_cells, avg_cells, _ = vModel.analyse_vertex_model()
            # Export excel with all_cells per file
            #all_cells.to_excel(os.path.join(output_folder, 'all_cells_%s.xlsx' % file))

            Geo = vModel.geo
            Set = vModel.set
            Set.currentT = 0

            # Initialize lists to store energies for the current file
            energy_lt_file = []
            energy_surface_file = []
            energy_volume_file = []
            energy_tri_ar_file = []
            neighbours_3d_file = []
            neighbours_apical_file = []
            neighbours_basal_file = []

            # Update measurements before computing energies
            Geo.update_measures()

            # Compute TriAR energy barrier
            kg_tri_ar = KgTriAREnergyBarrier(Geo)
            kg_tri_ar.compute_work(Geo, Set, None, False)

            # Compute the contractility
            kg_lt = KgContractility(Geo)
            kg_lt.compute_work(Geo, Set, None, False)

            # Compute Surface Tension
            kg_surface_area = KgSurfaceCellBasedAdhesion(Geo)
            kg_surface_area.compute_work(Geo, Set, None, False)

            # Compute Volume
            kg_volume = KgVolume(Geo)
            kg_volume.compute_work(Geo, Set, None, False)

            for cell_id in range(len(Geo.Cells)):
                if Geo.Cells[cell_id].AliveStatus is None:
                    continue

                energy_lt_file.append(kg_lt.energy_per_cell[cell_id])
                energy_surface_file.append(kg_surface_area.energy_per_cell[cell_id])
                energy_volume_file.append(kg_volume.energy_per_cell[cell_id])
                energy_tri_ar_file.append(kg_tri_ar.energy_per_cell[cell_id])

                # Scutoid detection
                apical_neigh = Geo.Cells[cell_id].compute_neighbours(location_filter='Top')
                basal_neigh = Geo.Cells[cell_id].compute_neighbours(location_filter='Bottom')

                if set(apical_neigh) != set(basal_neigh):
                    scutoid_cells.append({
                        'time': time,
                        'cell_id': cell_id,
                        'apical_neigh': len(apical_neigh),
                        'basal_neigh': len(basal_neigh)
                    })

                # Compute neighbours
                neighbours_3d = Geo.Cells[cell_id].compute_neighbours()
                neighbours_3d_file.append(len(neighbours_3d))
                neighbours_apical = Geo.Cells[cell_id].compute_neighbours(location_filter='Top')
                neighbours_apical_file.append(len(neighbours_apical))
                neighbours_basal = Geo.Cells[cell_id].compute_neighbours(location_filter='Bottom')
                neighbours_basal_file.append(len(neighbours_basal))

            # Track comprehensive metrics for all cells at this time step
            current_metrics = track_cell_metrics(vModel, Geo, kg_lt, kg_surface_area, kg_volume, kg_tri_ar, time)
            all_cell_metrics.append(current_metrics)

            # Append energies for the current file to the main lists
            energy_lt_cells.append(energy_lt_file)
            energy_surface_cells.append(energy_surface_file)
            energy_volume_cells.append(energy_volume_file)
            energy_tri_ar_cells.append(energy_tri_ar_file)
            neighbours_3d_cells.append(neighbours_3d_file)
            neighbours_apical_cells.append(neighbours_apical_file)
            neighbours_basal_cells.append(neighbours_basal_file)

            times.append(time)
            time += 1

    # Save all the information to a pkl file
    save_variables({'geometry_issues_all_times': geometry_issues_all_times,
                    'energy_lt_cells': energy_lt_cells,
                    'energy_surface_cells': energy_surface_cells,
                    'energy_volume_cells': energy_volume_cells,
                    'energy_tri_ar_cells': energy_tri_ar_cells,
                    'neighbours_3d_cells': neighbours_3d_cells,
                    'neighbours_apical_cells': neighbours_apical_cells,
                    'neighbours_basal_cells': neighbours_basal_cells,
                    'num_intercalations': num_intercalations,
                    'apical_flip_times': apical_flip_times,
                    'basal_flip_times': basal_flip_times,
                    'scutoid_cells': scutoid_cells,
                    'all_cell_metrics': all_cell_metrics,
                    'diagnostic_vtk_files': diagnostic_vtk_files,
                    'times': times,
                    }, '%s/megha_analysis.pkl' % output_folder)

# High energy cell detection
if energy_lt_cells and energy_surface_cells and energy_volume_cells:
    current_energies = [
        energy_lt_cells[-1][i] + energy_surface_cells[-1][i] + energy_volume_cells[-1][i]
        for i in range(len(energy_lt_cells[-1]))
    ]
    mean_energy = np.mean(current_energies)

    for cell_id, energy in enumerate(current_energies):
        if energy > (mean_energy * ENERGY_THRESHOLD_FACTOR):
            geometry_issues_all_times[-1]['high_energy_cells'].append({
                'cell_id': cell_id,
                'energy': energy,
                'mean_energy': mean_energy
            })

# Export diagnostic VTK
#export_path = os.path.join(output_folder, f'diagnostics_{time:04d}.vtk')
#export_diagnostic_vtk(Geo, export_path, time)
#diagnostic_vtk_files.append(export_path)

# # Custom cell geometry analysis
# weird_cells = []
#
# for cell_id in range(len(Geo.Cells)):
#     if Geo.Cells[cell_id].AliveStatus is None:
#         continue
#
#     neighs = Geo.Cells[cell_id].compute_neighbours()
#     if len(neighs) != 6:
#         print(f'Cell {cell_id} has {len(neighs)} neighbours')
#         print(f'Energy (LT): {kg_lt.energy_per_cell[cell_id]}')
#         print(f'Energy (Surface): {kg_surface_area.energy_per_cell[cell_id]}')
#         print(f'Energy (Volume): {kg_volume.energy_per_cell[cell_id]}')
#
#         weird_cells.append({
#             'id': cell_id,
#             'neighs': len(neighs),
#             'energy_lt': kg_lt.energy_per_cell[cell_id],
#             'energy_surface': kg_surface_area.energy_per_cell[cell_id],
#             'energy_volume': kg_volume.energy_per_cell[cell_id]
#         })
# Save the weird cells to an excel sheet
# weird_cells_df = pd.DataFrame(weird_cells)
# weird_cells_df.to_excel(os.path.join(output_folder, 'weird_cells.xlsx'), index=False)

# Combine all time steps
full_metrics_df = pd.concat(all_cell_metrics)

# Save the complete dataset
#full_metrics_df.to_excel(os.path.join(output_folder, 'full_cell_metrics.xlsx'), index=False)

# Identify persistent high-energy cells with respect the last two time steps
#mean_energy = full_metrics_df['total_energy'].mean()
#high_energy_cells = full_metrics_df[full_metrics_df['total_energy'] > mean_energy * ENERGY_THRESHOLD_FACTOR]
energy_trend = all_cell_metrics[-1]['total_energy'] / all_cell_metrics[-2]['total_energy']
high_energy_cells = full_metrics_df[np.isin(full_metrics_df.cell_id, all_cell_metrics[-1][energy_trend > 1.3]['cell_id'])]
high_energy_ids = high_energy_cells['cell_id'].unique()

# Define pre-spike window (Δt=5)
At = 2
cases = []
controls = []
t = np.max(high_energy_cells['time'])

for cell_id in all_cell_metrics[-1]['cell_id'].unique():
    current_cell = full_metrics_df[full_metrics_df['cell_id'] == cell_id]
    case_features = current_cell.iloc[t - 2 * At:t - At].mean().to_dict()  # Baseline
    case_features.update(current_cell.iloc[t - At:t].std().add_prefix('volatility_'))  # Trends
    if cell_id in high_energy_ids:
        # Case: Pre-spike features
        cases.append(case_features)
    else:
        # Control: Random non-spike window (matched to time phase)
        controls.append(case_features)

# Create DataFrames
df_cases = pd.DataFrame(cases).assign(label=1)
df_controls = pd.DataFrame(controls).assign(label=0)
df_combined = pd.concat([df_cases, df_controls])

# Statistical tests
results = []
for feature in df_combined.columns.drop('label'):
    t_stat, p_val = stats.mannwhitneyu(
        df_cases[feature],
        df_controls[feature],
        alternative='two-sided'
    )
    results.append({'feature': feature, 'p_val': p_val})

# Correct for multiple testing
df_results = pd.DataFrame(results)
df_results['p_val_adj'] = stats.false_discovery_control(df_results['p_val'])

# Plot top features
top_features = df_results.loc[df_results['p_val_adj'] < 0.01, 'feature']
# Remove 'volatility_' prefix
top_features = top_features.str.replace('volatility_', '', regex=False)
# Unique feature names
top_features = top_features.unique()
plt.figure()
plt.plot(full_metrics_df.loc[high_energy_ids, top_features], label=top_features)
#plt.scatter(high_energy_ids, full_metrics_df.loc[high_energy_ids, top_features], c='red', label='Spike')
plt.legend()
plt.show()

# Save high energy cell IDs
def plot_metric_comparison(metric):
    """Compare evolution of a metric between high-energy and normal cells"""
    plt.figure(figsize=(10, 6))

    # Generate a color map for unique cell IDs
    unique_ids = high_energy_ids
    colors = cm.get_cmap('tab20', len(unique_ids))

    # Plot high-energy cells individually with labels and unique colors
    for i, cell_id in enumerate(unique_ids):
        cell_data = full_metrics_df[full_metrics_df['cell_id'] == cell_id]
        plt.plot(cell_data['time'], cell_data[metric], marker='o',
                 color=colors(i),
                 label=f'High Energy Cell {cell_id}',
                 linewidth=1.5)

    # Plot average of normal cells
    normal_avg = full_metrics_df[~full_metrics_df['cell_id'].isin(high_energy_ids)].groupby('time')[
        metric].mean()
    plt.plot(normal_avg.index, normal_avg.values, 'k--', marker='o', linewidth= 1 , label='Normal Cells Avg')

    plt.title(f'Comparison of {metric} Evolution')
    plt.xlabel('Time')
    plt.ylabel(metric)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'{metric}_comparison.png'))
    plt.close()


# ====== PLOTTING SECTION (AFTER ALL DATA COLLECTION) ======

pd.DataFrame(high_energy_cells).to_excel(os.path.join(output_folder, 'high_energy_cell_ids.xlsx'),
                                                    index=False)


def plot_energy_around_t1(cell_id):
    """Plot energy components before/during/after T1 transition"""
    cell_data = full_metrics_df[full_metrics_df['cell_id'] == cell_id]

    plt.figure(figsize=(10, 6))
    plt.stackplot(cell_data['time'],
                  cell_data['energy_lt'],
                  cell_data['energy_surface'],
                  cell_data['energy_volume'],
                  cell_data['energy_tri_ar'],
                  labels=['Line Tension', 'Surface', 'Volume', 'TriAR'])
    plt.title(f'Energy Composition for Cell {cell_id} Around T1 Transition')
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.legend()
    plt.savefig(os.path.join(output_folder, f'cell_{cell_id}_t1_energy.png'))
    plt.close()

# 1. Total Energies Plot
plt.figure()
plt.plot(times, [sum(energy) for energy in energy_lt_cells], label=f'Line tension {Set.cLineTension}')
plt.plot(times, [sum(energy) for energy in energy_surface_cells],
         label=f'Surface tension apical {Set.lambdaS1}, basal {Set.lambdaS2}, lateral {Set.lambdaS3}')
plt.plot(times, [sum(energy) for energy in energy_volume_cells], label=f'Volume {Set.lambdaV}')
plt.plot(times, [sum(energy) for energy in energy_tri_ar_cells], label=f'TriAR energy barrier {Set.lambdaR}')
plt.legend()
plt.savefig(os.path.join(output_folder, 'total_energies.png'))
plt.close()

# 2. Neighbor Analysis
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.plot(times, [np.mean(n) for n in neighbours_3d_cells], label='3D neighbours')
plt.plot(times, [np.mean(n) for n in neighbours_apical_cells], label='Apical neighbours')
plt.plot(times, [np.mean(n) for n in neighbours_basal_cells], label='Basal neighbours')
plt.legend()
plt.title("Average Neighbors")

plt.subplot(122)
plt.plot(times, [np.std(n) for n in neighbours_3d_cells], label='3D neighbours')
plt.plot(times, [np.std(n) for n in neighbours_apical_cells], label='Apical neighbours')
plt.plot(times, [np.std(n) for n in neighbours_basal_cells], label='Basal neighbours')
plt.legend()
plt.title("Neighbor Variability")
plt.savefig(os.path.join(output_folder, 'neighbour_analysis.png'))
plt.close()

# 3. Intercalation Analysis


if len(num_intercalations) > 0:
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.plot(times, num_intercalations, 'o-')
    plt.title("T1 Intercalation Events")
    plt.xlabel("Time")
    plt.ylabel("Count")

    plt.subplot(132)
    if len(apical_flip_times) == len(times):
        plt.plot(times, apical_flip_times, 'o-', label='Apical')
        plt.plot(times, basal_flip_times, 'o-', label='Basal')
        plt.legend()
        plt.title("Layer-Specific Flips")

    plt.subplot(133)
    plt.plot([sum(e) for e in energy_lt_cells], num_intercalations, 'o')
    plt.xlabel("Total Contractility Energy")
    plt.ylabel("Intercalations")
    plt.title("Energy vs Intercalations")

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'intercalation_analysis.png'))
    plt.close()

# 4. Energy Composition
if len(energy_lt_cells) > 0:
    total_energy = np.array([sum(e) for e in energy_lt_cells]) + \
                   np.array([sum(e) for e in energy_surface_cells]) + \
                   np.array([sum(e) for e in energy_volume_cells]) + \
                   np.array([sum(e) for e in energy_tri_ar_cells])

    plt.figure()
    plt.stackplot(times,
                  [sum(e) / te for e, te in zip(energy_lt_cells, total_energy)],
                  [sum(e) / te for e, te in zip(energy_surface_cells, total_energy)],
                  [sum(e) / te for e, te in zip(energy_volume_cells, total_energy)],
                  [sum(e) / te for e, te in zip(energy_tri_ar_cells, total_energy)],
                  labels=['Line Tension', 'Surface', 'Volume', 'TriAR'])
    plt.legend(loc='upper left')
    plt.title("Energy Composition Over Time")
    plt.savefig(os.path.join(output_folder, 'energy_composition.png'))
    plt.close()

# 5. Individual Cell Analysis (keep this in the loop but add proper closing)
for cell_id in range(len(Geo.Cells)):
    if Geo.Cells[cell_id].AliveStatus is None or cell_id in Geo.BorderCells:
        continue

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    # Energy plot
    ax1.plot(times, [e[cell_id] for e in energy_lt_cells], label='Line tension')
    ax1.plot(times, [e[cell_id] for e in energy_surface_cells], label='Surface')
    ax1.plot(times, [e[cell_id] for e in energy_volume_cells], label='Volume')
    ax1.plot(times, [e[cell_id] for e in energy_tri_ar_cells], label='TriAR')
    ax1.legend()

    # Neighbor plot
    ax2.plot(times, [n[cell_id] for n in neighbours_3d_cells], label='3D')
    ax2.plot(times, [n[cell_id] for n in neighbours_apical_cells], label='Apical')
    ax2.legend()

    plt.savefig(os.path.join(output_folder, f'cell_{cell_id}_analysis.png'))
    plt.close()

# Geometry diagnostics summary plot
plt.figure(figsize=(15, 10))
plt.suptitle('Geometry Quality Diagnostics')

# Subplot 1: Aspect ratios
plt.subplot(2, 2, 1)
max_aspect_3d = [max([c['aspect_3d'] for c in t['distorted_cells']], default=1)
                 for t in geometry_issues_all_times]
plt.plot(times, max_aspect_3d, 'r-')
plt.axhline(ASPECT_RATIO_3D_THRESHOLD, color='r', linestyle='--')
plt.title('Max 3D Aspect Ratio')
plt.xlabel('Time')
plt.ylabel('Ratio')

# Subplot 2: Issue counts
plt.subplot(2, 2, 2)
plt.plot(times, [len(t['distorted_cells']) for t in geometry_issues_all_times], 'r-', label='Distorted')
plt.plot(times, [len(t['inverted_cells']) for t in geometry_issues_all_times], 'b-', label='Inverted')
plt.plot(times, [len(t['small_angle_cells']) for t in geometry_issues_all_times], 'g-', label='Small Angles')
plt.title('Issue Counts')
plt.xlabel('Time')
plt.ylabel('Number of Cells')
plt.legend()

# Subplot 3: Energy correlation
plt.subplot(2, 2, 3)
plt.plot(times, [len(t['high_energy_cells']) for t in geometry_issues_all_times], 'm-')
plt.title('High Energy Cells')
plt.xlabel('Time')
plt.ylabel('Count')

# Subplot 4: Minimum angles
plt.subplot(2, 2, 4)
min_angles = [min([c['min_angle'] for c in t['small_angle_cells']], default=90)
              for t in geometry_issues_all_times]
plt.plot(times, min_angles, 'g-')
plt.axhline(MIN_ANGLE_THRESHOLD, color='g', linestyle='--')
plt.title('Minimum Face Angle')
plt.xlabel('Time')
plt.ylabel('Degrees')

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'geometry_diagnostics_summary.png'))
plt.close()

# Right after your existing plotting code, add:

# Generate plots for high-energy cells around T1 transitions
for cell_id in high_energy_ids:  # Just plot first 5 as example
    plot_energy_around_t1(cell_id)

# Generate comparison plots for key metrics
for metric in ['aspect_3d', 'min_angle', 'total_energy', 'neighbours_basal']:
    plot_metric_comparison(metric)

# Create energy phase diagram
plt.figure(figsize=(10, 8))
sc = plt.scatter(full_metrics_df['aspect_3d'],
                 full_metrics_df['min_angle'],
                 c=full_metrics_df['total_energy'],
                 cmap='viridis', alpha=0.6)
plt.colorbar(sc, label='Total Energy')
plt.axvline(ASPECT_RATIO_3D_THRESHOLD, color='r', linestyle='--')
plt.axhline(MIN_ANGLE_THRESHOLD, color='r', linestyle='--')
plt.xlabel('3D Aspect Ratio')
plt.ylabel('Minimum Angle (degrees)')
plt.title('Energy Landscape vs Geometric Parameters')
plt.savefig(os.path.join(output_folder, 'energy_phase_diagram.png'))
plt.close()

# Correlation analysis
breakdown_corr = full_metrics_df.corr()['total_energy'].sort_values(ascending=False)
print("Parameters most correlated with high energy:")
print(breakdown_corr)
pd.DataFrame(breakdown_corr).to_excel(os.path.join(output_folder, 'energy_correlations.xlsx'))

# Export detailed diagnostics
diagnostics_df = pd.DataFrame([
    {**issue, 'time': t['time']}
    for t in geometry_issues_all_times
    for issue_type in ['distorted_cells', 'inverted_cells', 'small_angle_cells', 'high_energy_cells']
    for issue in t[issue_type]
])
diagnostics_df.to_excel(os.path.join(output_folder, 'geometry_diagnostics.xlsx'), index=False)

