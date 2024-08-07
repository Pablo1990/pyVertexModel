import numpy as np

from src.pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage import VertexModelVoronoiFromTimeImage
from src.pyVertexModel.util.utils import load_state


def analyse_edge_recoil(v_model, n_ablations=1, location_filter=0):
    """
    Analyse how much an edge recoil if we ablate an edge of a cell
    :param v_model:
    :return:
    """
    possible_cells_to_ablate = [cell for cell in v_model.geo.Cells if cell.AliveStatus == 1 and cell.ID not
                                in v_model.geo.BorderCells]
    recoil = np.zeros(n_ablations)
    for i in range(n_ablations):
        # Cells to ablate
        #cell_to_ablate = np.random.choice(possible_cells_to_ablate, 1)
        cell_to_ablate = [v_model.geo.Cells[0]]

        # Pick the neighbouring cell to ablate
        neighbours = cell_to_ablate[0].compute_neighbours(location_filter)
        possible_neighbours = [neighbour for neighbour in neighbours if neighbour in possible_cells_to_ablate]
        neighbour_to_ablate = np.random.choice(neighbours, 1)

        # Pick the neighbour and put it in the list
        cells_to_ablate = [cell_to_ablate[0].ID, neighbour_to_ablate[0]]

        # Get the edge that share both cells
        edge_length_init = get_edge_length(cells_to_ablate, location_filter, v_model)

        # Ablate the edge
        v_model.set.ablation = True
        v_model.geo.cellsToAblate = cells_to_ablate
        v_model.set.TInitAblation = v_model.t
        v_model.geo.ablate_cells(v_model.set, v_model.t, combine_cells=False)

        # Relax the system
        v_model.set.tend = v_model.t + 1
        v_model.set.ablation = False
        v_model.iterate_over_time()

        # Get the edge length
        edge_length_final = get_edge_length(cells_to_ablate, location_filter, v_model)

        # Calculate the recoil
        recoil[i] = (edge_length_final - edge_length_init) / edge_length_init

    return np.mean(recoil)


def get_edge_length(cells_to_ablate, location_filter, v_model):
    """
    Get the edge length of the edge that share the cells_to_ablate
    :param cells_to_ablate:
    :param location_filter:
    :param v_model:
    :return:
    """

    vertices = []
    cell = [cell for cell in v_model.geo.Cells if cell.ID == cells_to_ablate[0]][0]
    for c_face in cell.Faces:
        if c_face.InterfaceType == location_filter:
            for c_tri in c_face.Tris:
                if np.all(np.isin(cells_to_ablate, c_tri.SharedByCells)):
                    vertices.append(cell.Y[c_tri.Edge[0]])
                    vertices.append(cell.Y[c_tri.Edge[1]])
    # Get the edge length
    edge_length_init = 0
    for num_vertex in range(0, len(vertices), 2):
        edge_length_init += np.linalg.norm(vertices[num_vertex] - vertices[num_vertex + 1])

    return edge_length_init


v_model = VertexModelVoronoiFromTimeImage()
output_folder = v_model.set.OutputFolder
load_state(v_model,
           '/media/pablo/d7c61090-024c-469a-930c-f5ada47fb049/PabloVicenteMunuera/VertexModel/pyVertexModel/Result/'
           '08-07_114514__Cells_150_visc_100_lVol_1_kSubs_1_lt_0.001_ltExt_0.001_noise_0_refA0_1_eTriAreaBarrier_0_eARBarrier_0_RemStiff_0.95_lS1_4_lS2_0.4_lS3_4_pString_0.006'
           '/data_step_300.pkl')
v_model.set.OutputFolder = output_folder
print(analyse_edge_recoil(v_model, n_ablations=1))
