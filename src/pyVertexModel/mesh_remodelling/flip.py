import logging
from itertools import combinations

import networkx as nx
import numpy as np
from scipy.spatial import Delaunay

from src.pyVertexModel.geometry.geo import edge_valence_t
from src.pyVertexModel.util.utils import ismember_rows

logger = logging.getLogger("pyVertexModel")


def post_flip(Tnew, Ynew, oldTets, Geo, Geo_n, Geo_0, Dofs, Set, old_geo):
    """
    Post flip
    :param Tnew:
    :param Ynew:
    :param oldTets:
    :param Geo:
    :param Geo_n:
    :param Geo_0:
    :param Dofs:
    :param Set:
    :param old_geo:
    :return:
    """
    Geo.add_and_rebuild_cells(old_geo, oldTets, Tnew, Ynew, Set, True)
    has_converged = True

    return Geo_0, Geo_n, Geo, Dofs, has_converged


def do_flip32(Y, X12):
    """
    Do flip 32
    :param Y:
    :param X12:
    :return:
    """
    min_length = np.min([np.linalg.norm(Y[0] - Y[1]), np.linalg.norm(Y[2] - Y[1]), np.linalg.norm(Y[0] - Y[2])])
    perpend = np.cross(Y[0] - Y[1], Y[2] - Y[1])
    n_perpen = perpend / np.linalg.norm(perpend)
    center = np.sum(Y, axis=0) / 3
    Nx = X12[0] - center
    Nx = Nx / np.linalg.norm(Nx)
    if np.dot(n_perpen, Nx) > 0:
        Y1 = center + (min_length * n_perpen)
        Y2 = center - (min_length * n_perpen)
    else:
        Y1 = center - (min_length * n_perpen)
        Y2 = center + (min_length * n_perpen)
    Yn = np.vstack((Y1, Y2))
    return Yn


def y_flip32(Ys, Ts, YsToChange, Geo):
    """
    Y flip 32
    :param Ys:
    :param Ts:
    :param YsToChange:
    :param Geo:
    :return:
    """
    n = list(set(Ts[YsToChange[0]]).intersection(Ts[YsToChange[1]], Ts[YsToChange[2]]))
    N = np.unique(Ts[YsToChange])  # all nodes
    N = N[~np.isin(N, n)]

    N3 = N[~np.isin(N, Ts[YsToChange[0]])]
    Tnew1 = Ts[YsToChange[0]].copy()
    Tnew2 = Tnew1.copy()
    Tnew1[np.isin(Ts[YsToChange[0]], n[1])] = N3
    Tnew2[np.isin(Ts[YsToChange[0]], n[0])] = N3
    Tnew = np.vstack((Tnew1, Tnew2))

    # The new vertices
    Xs = np.zeros((len(n), 3))
    for ni in range(len(n)):
        Xs[ni, :] = Geo.Cells[n[ni]].X
    Ynew = do_flip32(Ys[YsToChange], Xs)

    return Ynew, Tnew


def do_flip23(Yo, Geo, n3):
    """
    Do flip 23
    :param Yo:
    :param Geo:
    :param n3:
    :return:
    """
    # the new vertices are placed at a distance "Length of the line to be
    # removed" from the "center of the line to be removed" in the direction of
    # the barycenter of the corresponding tet

    # Center and Length of The line to be removed
    length = np.linalg.norm(Yo[0] - Yo[1])
    center = np.sum(Yo, axis=0) / 2

    # Strategy Number 2
    center2 = sum([Geo.Cells[n].X for n in n3]) / 3

    direction = np.zeros((3, 3))
    n3 = np.append(n3, n3[0])
    for numCoord in range(3):
        node1 = (Geo.Cells[n3[numCoord]].X + Geo.Cells[n3[numCoord + 1]].X) / 2
        direction[numCoord, :] = node1 - center2
        direction[numCoord, :] = direction[numCoord, :] / np.linalg.norm(direction[numCoord, :])

    Yn = np.array([center + direction[0, :] * length,
                   center + direction[1, :] * length,
                   center + direction[2, :] * length])

    return Yn


def y_flip23(Ys, Ts, YsToChange, Geo):
    """
    Y flip 23
    :param Ys:
    :param Ts:
    :param YsToChange:
    :param Geo:
    :return:
    """
    n3 = Ts[YsToChange[0]][np.isin(Ts[YsToChange[0]], Ts[YsToChange[1]])]
    n1 = Ts[YsToChange[0]][~np.isin(Ts[YsToChange[0]], n3)]
    n2 = Ts[YsToChange[1]][~np.isin(Ts[YsToChange[1]], n3)]
    num = np.array([0, 1, 2, 3])
    num = num[Ts[YsToChange[0]] == n1]
    if num == 2 or num == 4:
        Tnew = np.block([[n3[0], n3[1], n2, n1],
                         [n3[1], n3[2], n2, n1],
                         [n3[2], n3[0], n2, n1]])
    else:
        Tnew = np.block([[n3[0], n3[1], n1, n2],
                         [n3[1], n3[2], n1, n2],
                         [n3[2], n3[0], n1, n2]])

    ghostNodes = np.isin(Tnew, Geo.XgID)
    ghostNodes = np.all(ghostNodes, axis=1)

    Ynew = do_flip23(Ys[YsToChange], Geo, n3)
    Ynew = Ynew[~ghostNodes]

    return Ynew, Tnew


def y_flip_nm_recursive(TOld, TRemoved, Tnew, Ynew, oldYs, Geo, possibleEdges, XsToDisconnect, treeOfPossibilities,
                        parentNode, arrayPos):
    """
    Y flip NM recursive
    :param TOld:
    :param TRemoved:
    :param Tnew:
    :param Ynew:
    :param oldYs:
    :param Geo:
    :param possibleEdges:
    :param XsToDisconnect:
    :param treeOfPossibilities:
    :param parentNode:
    :param arrayPos:
    :return:
    """
    endNode = 1

    Told_original = TOld.copy()
    if TOld.shape[0] == 3:
        Ynew_c, Tnew_c = y_flip32(oldYs, TOld, [0, 1, 2], Geo)
        TRemoved.insert(arrayPos, TOld)
        Tnew.insert(arrayPos, Tnew_c)
        Ynew.insert(arrayPos, Ynew_c)
        treeOfPossibilities.add_edge(parentNode, arrayPos)
        treeOfPossibilities.add_edge(arrayPos, endNode)
        arrayPos += 1
    else:
        # https://link.springer.com/article/10.1007/s00366-016-0480-z#Fig2
        for numPair in range(possibleEdges.shape[0]):
            valence, sharedTets, tetIds = edge_valence_t(Told_original, possibleEdges[numPair, :])

            # Valence == 1, is an edge that can be removed.
            # Valence == 2, a face can be removed.
            if valence == 2:
                Ynew_23, Tnew_23 = y_flip23(oldYs, Told_original, tetIds, Geo)

                TRemoved.insert(arrayPos, Told_original[tetIds, :])
                Tnew.insert(arrayPos, Tnew_23)
                Ynew.insert(arrayPos, Ynew_23)
                treeOfPossibilities.add_edge(parentNode, arrayPos)

                TOld = Told_original.copy()
                # Remove the row of tets that are associated to that edgeToDisconnect
                TOld = np.delete(TOld, tetIds, axis=0)
                TOld = np.vstack((TOld, Tnew_23))

                # Update and get the tets that are associated to that edgeToDisconnect
                # Valence should have decreased
                _, TOld_new, _ = edge_valence_t(TOld, XsToDisconnect)
                [Ynew, Tnew, TRemoved, treeOfPossibilities, arrayPos] = y_flip_nm_recursive(TOld_new, TRemoved, Tnew,
                                                                                            Ynew, oldYs, Geo,
                                                                                            possibleEdges,
                                                                                            XsToDisconnect,
                                                                                            treeOfPossibilities,
                                                                                            arrayPos,
                                                                                            arrayPos + 1)

    return Ynew, Tnew, TRemoved, treeOfPossibilities, arrayPos


def compute_tet_volume(tet, Geo):
    """
    Compute the volume of a tetrahedron
    :param tet:
    :param Geo:
    :return:
    """
    # Get the coordinates of the tetrahedron
    Xs = np.vstack([Geo.Cells[t].X for t in tet])

    # Get delaunay triangulation of the 4 nodes
    tri = Delaunay(Xs)
    Xs = Xs[tri.simplices[0], :]

    # Vector from the first node to the other three
    y1 = Xs[1, :] - Xs[0, :]
    y2 = Xs[2, :] - Xs[0, :]
    y3 = Xs[3, :] - Xs[0, :]

    # Compute the volume
    Ytri = np.array([y1, y2, y3])
    vol = abs(np.linalg.det(Ytri)) / 6

    return vol


def get_4_fold_tets(Geo):
    """
    Get the tets that have 4-fold nodes without debris cells
    :param Geo:
    :return:
    """
    allTets = np.vstack([cell.T for cell in Geo.Cells if cell.AliveStatus is not None])

    ghostNodesWithoutDebris = np.setdiff1d(Geo.XgID, Geo.RemovedDebrisCells)
    ghostNodesWithoutDebris = np.setdiff1d(ghostNodesWithoutDebris, Geo.BorderGhostNodes)

    tets = allTets[np.all(~np.isin(allTets, ghostNodesWithoutDebris), axis=1)]
    tets = np.unique(tets, axis=0)

    return tets


def y_flip_nm(old_tets, cell_to_intercalate_with, old_ys, xs_to_disconnect, Geo, Set, cell_to_split_from):
    """
    Y flip NM
    :param old_tets:
    :param cell_to_intercalate_with:
    :param old_ys:
    :param xs_to_disconnect:
    :param Geo:
    :param Set:
    :return:
    """
    Xs_gToDisconnect = xs_to_disconnect[np.isin(xs_to_disconnect, Geo.XgID)]

    Geo_without_scutoids = Geo.copy()

    # Temporary remove 4-cell tetrahedra
    tets4_cells = get_4_fold_tets(Geo_without_scutoids)
    ys_4_cells = Geo_without_scutoids.remove_tetrahedra(tets4_cells)
    # tets4_cells = np.unique(np.sort(tets4_cells, axis=1), axis=0)
    ghost_nodes_without_debris = np.setdiff1d(Geo_without_scutoids.XgID, Geo_without_scutoids.RemovedDebrisCells)

    Xs = np.unique(old_tets)
    Xs_c = Xs[~np.isin(Xs, ghost_nodes_without_debris)]
    intercalation_flip = 0
    if len(Xs_c) > 3:
        intercalation_flip = 1

    # Step 1: Keep the boundary of tets not changed.
    boundaryNodes = np.unique(old_tets)

    # Step 2: Get edges that can be added when removing the other one
    possibleEdges = np.vstack(list(combinations(boundaryNodes, 2)))

    # Step 3: Select the edge to add
    edgeToConnect = np.block([np.array(cell_to_intercalate_with), Xs_gToDisconnect])
    possibleEdges = possibleEdges[~np.all(np.isin(possibleEdges, edgeToConnect), axis=1)]

    # Step 4: Propagate the change to get the remaining tets
    # Based on https://dl.acm.org/doi/pdf/10.1145/2629697

    treeOfPossibilities = nx.DiGraph()
    treeOfPossibilities.add_node(2)
    TRemoved = [None, None]
    Tnew = [None, None]
    Ynew = [None, None]
    parentNode = 0
    arrayPos = 2
    endNode = 1

    # Re-order old_tets randomly to avoid bias
    np.random.shuffle(old_tets)

    _, Tnew, TRemoved, treeOfPossibilities, _ = y_flip_nm_recursive(old_tets, TRemoved, Tnew, Ynew, old_ys, Geo_without_scutoids,
                                                                    possibleEdges,
                                                                    xs_to_disconnect, treeOfPossibilities, parentNode,
                                                                    arrayPos)

    # Step 5: For each combination of edges to remove, check if it is valid
    logger.info(f"Number of possible combinations: {len(Tnew)}")
    new_tets_tree, Geo_new = get_best_new_tets_combination(Geo, Set, TRemoved, Tnew, Xs, endNode,
                                                           ghost_nodes_without_debris, intercalation_flip, old_tets,
                                                           parentNode, tets4_cells, treeOfPossibilities,
                                                           xs_to_disconnect, cell_to_split_from)

    # Get the last combination from new_tets_tree
    Tnew = new_tets_tree
    Ynew = []
    return Tnew, Ynew, Geo_new


def dfs(graph, start, end):
    """
    Depth first search
    :param graph:
    :param start:
    :param end:
    :return:
    """
    stack = [(start, [start])]
    while stack:
        (node, path) = stack.pop()
        for next_node in set(graph[node]) - set(path):
            if next_node == end:
                yield path + [next_node]
            else:
                stack.append((next_node, path + [next_node]))


def get_best_new_tets_combination(Geo, Set, TRemoved, Tnew, Xs, endNode, ghost_nodes_without_debris, intercalation_flip,
                                  old_tets, parentNode, tets4_cells, treeOfPossibilities, xs_to_disconnect,
                                  cell_to_split_from):
    """
    Get the best combination of new tets
    :param Geo:
    :param Set:
    :param TRemoved:
    :param Tnew:
    :param Xs:
    :param endNode:
    :param ghost_nodes_without_debris:
    :param intercalation_flip:
    :param old_tets:
    :param parentNode:
    :param tets4_cells:
    :param treeOfPossibilities:
    :return:
    """
    Geo_final = Geo
    new_tets_tree = None
    valence_segment = np.inf

    xs_to_disconnect_cells = xs_to_disconnect[~np.isin(xs_to_disconnect, Geo.XgID)]

    for c_path in dfs(treeOfPossibilities, parentNode, endNode):
        c_path = np.array(c_path)
        new_tets = np.vstack(old_tets)

        for posPath in c_path[c_path > 1]:
            toAdd = np.sort(Tnew[posPath])
            toRemove = TRemoved[posPath]
            new_tets = new_tets[~ismember_rows(np.sort(new_tets, 1), np.sort(toRemove, 1))[0]]
            new_tets = np.vstack((new_tets, toAdd))

        if np.all(np.sum(np.isin(new_tets, Geo.XgID), axis=1) < 4):
            if intercalation_flip:
                Xs_c = Xs[~np.isin(Xs, ghost_nodes_without_debris)]
                # Split into the different combination of tets using 'combinations'
                for combination in combinations(Xs_c, 4):
                    if ~ismember_rows(np.array(combination), np.vstack([new_tets, tets4_cells]))[0][0]:
                        new_tets = np.append(new_tets, np.array([combination]), axis=0)

            current_valence_segment, shared_tets, _ = (
                edge_valence_t(new_tets, [xs_to_disconnect_cells, cell_to_split_from]))
            if current_valence_segment < valence_segment:
                try:
                    Geo_new = Geo.copy()
                    Geo_new.remove_tetrahedra(old_tets)
                    Geo_new.add_tetrahedra(Geo, new_tets, None, Set)
                    Geo_new.rebuild(Geo, Set, cells_to_rebuild=np.unique(new_tets))
                    Geo_new.build_global_ids()

                    new_tets_tree = new_tets
                    Geo_final = Geo_new
                    valence_segment = current_valence_segment
                    logger.info(f"New combination found with valence segment: {valence_segment}")
                except Exception as ex:
                    logger.warning(f"Exception on flip remodelling: {ex}")
    return new_tets_tree, Geo_final


def add_new_info(Tnew_23, Ynew_23, final_tets, final_ys, new_tets, new_ys, removed_tets, removed_ys, tetIds):
    """
    Add new info
    :param Tnew_23:
    :param Ynew_23:
    :param final_tets:
    :param final_ys:
    :param new_tets:
    :param new_ys:
    :param removed_tets:
    :param removed_ys:
    :param tetIds:
    :return:
    """
    new_tets = np.append(new_tets, Tnew_23)
    new_ys = np.append(new_ys, Ynew_23)
    removed_tets = np.append(removed_tets, final_tets[tetIds, :])
    removed_ys = np.append(removed_ys, final_ys[tetIds, :])
    return new_tets, new_ys, removed_tets, removed_ys


def update_test_ys(Tnew_23, Ynew_23, final_tets, final_ys, tetIds):
    """
    Update the tets and ys after a 2-3 flip
    :param Tnew_23:
    :param Ynew_23:
    :param final_tets:
    :param final_ys:
    :param tetIds:
    :return:
    """
    # Update and get the tets that are associated to that edgeToDisconnect
    final_tets = np.delete(final_tets, tetIds, axis=0)
    final_tets = np.vstack((final_tets, Tnew_23))
    # Update and get the Ys that are associated to that edgeToDisconnect
    final_ys = np.delete(final_ys, tetIds, axis=0)
    final_ys = np.vstack((final_ys, Ynew_23))
    return final_tets, final_ys
