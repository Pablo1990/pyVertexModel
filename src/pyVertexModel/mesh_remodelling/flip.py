import logging
from itertools import combinations

import networkx as nx
import numpy as np
from scipy.spatial import Delaunay

from src.pyVertexModel.geometry.geo import edgeValenceT
from src.pyVertexModel.mesh_remodelling.remodelling import solve_remodeling_step
from src.pyVertexModel.util.utils import ismember_rows

logger = logging.getLogger("pyVertexModel")


def post_flip(Tnew, Ynew, oldTets, Geo, Geo_n, Geo_0, Dofs, newYgIds, Set, flipName, segmentToChange):
    """
    Summary of this function goes here
    Detailed explanation goes here
    """

    hasConverged = 0
    Geo_backup = Geo.copy()
    Geo_n_backup = Geo_n.copy()
    Geo_0_backup = Geo_0.copy()
    Dofs_backup = Dofs.copy()

    logger.info(f"{flipName}-Flip: {segmentToChange[0]} {segmentToChange[1]}.")

    Geo.add_and_rebuild_cells(Geo.copy(), oldTets, Tnew, Ynew, Set, 1)
    Geo_n = Geo.copy()

    Dofs.get_dofs(Geo, Set)
    Dofs, Geo = Dofs.get_remodel_dofs(Tnew, Dofs, Geo)
    Geo, Set, DidNotConverge = solve_remodeling_step(Geo_0, Geo_n, Geo, Dofs, Set)
    if DidNotConverge:
        Geo = Geo_backup
        Geo_n = Geo_n_backup
        Geo_0 = Geo_0_backup
        Dofs = Dofs_backup
        logger.info(f"{flipName}-Flip rejected: did not converge")
        return Geo_0, Geo_n, Geo, Dofs, newYgIds, hasConverged

    Geo.update_measures()

    newYgIds = list(set(newYgIds + Geo.AssemblegIds))

    hasConverged = 1

    return Geo_0, Geo_n, Geo, Dofs, newYgIds, hasConverged


def DoFlip32(Y, X12):
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


def YFlip32(Ys, Ts, YsToChange, Geo):
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
    Ynew = DoFlip32(Ys[YsToChange], Xs)

    return Ynew, Tnew


def DoFlip23(Yo, Geo, n3):
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


def YFlip23(Ys, Ts, YsToChange, Geo):
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

    Ynew = DoFlip23(Ys[YsToChange], Geo, n3)
    Ynew = Ynew[~ghostNodes]

    return Ynew, Tnew


def YFlipNM_recursive(TOld, TRemoved, Tnew, Ynew, oldYs, Geo, possibleEdges, XsToDisconnect, treeOfPossibilities,
                      parentNode, arrayPos):
    endNode = 1

    Told_original = TOld
    if TOld.shape[0] == 3:
        Ynew_c, Tnew_c = YFlip32(oldYs, TOld, [0, 1, 2], Geo)
        TRemoved.insert(arrayPos, TOld)
        Tnew.insert(arrayPos, Tnew_c)
        Ynew.insert(arrayPos, Ynew_c)
        treeOfPossibilities.add_edge(parentNode, arrayPos)
        treeOfPossibilities.add_edge(arrayPos, endNode)
        arrayPos += 1
    else:
        # https://link.springer.com/article/10.1007/s00366-016-0480-z#Fig2
        for numPair in range(possibleEdges.shape[0]):
            valence, sharedTets, tetIds = edgeValenceT(Told_original, possibleEdges[numPair, :])

            # Valence == 1, is an edge that can be removed.
            # Valence == 2, a face can be removed.
            if valence == 2:
                Ynew_23, Tnew_23 = YFlip23(oldYs, Told_original, tetIds, Geo)

                TRemoved.insert(arrayPos, Told_original[tetIds, :])
                Tnew.insert(arrayPos, Tnew_23)
                Ynew.insert(arrayPos, Ynew_23)
                treeOfPossibilities.add_edge(parentNode, arrayPos)

                TOld = Told_original
                # Remove the row of tets that are associated to that edgeToDisconnect
                TOld = np.delete(TOld, tetIds, axis=0)
                TOld = np.vstack((TOld, Tnew_23))

                # Update and get the tets that are associated to that edgeToDisconnect
                # Valence should have decreased
                _, TOld_new, _ = edgeValenceT(TOld, XsToDisconnect)
                [Ynew, Tnew, TRemoved, treeOfPossibilities, arrayPos] = YFlipNM_recursive(TOld_new, TRemoved, Tnew,
                                                                                          Ynew, oldYs, Geo,
                                                                                          possibleEdges,
                                                                                          XsToDisconnect,
                                                                                          treeOfPossibilities, arrayPos,
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
    allTets = np.vstack([cell.T for cell in Geo.Cells])

    ghostNodesWithoutDebris = np.setdiff1d(Geo.XgID, Geo.RemovedDebrisCells)

    tets = allTets[np.all(~np.isin(allTets, ghostNodesWithoutDebris), axis=1)]
    tets = np.unique(tets, axis=0)

    return tets


def YFlipNM(old_tets, cell_to_intercalate_with, old_ys, xs_to_disconnect, Geo, Set):
    """

    :param old_tets:
    :param cell_to_intercalate_with:
    :param old_ys:
    :param xs_to_disconnect:
    :param Geo:
    :param Set:
    :return:
    """
    Xs_gToDisconnect = xs_to_disconnect[np.isin(xs_to_disconnect, Geo.XgID)]

    # Temporary remove 4-cell tetrahedra
    tets4_cells = get_4_fold_tets(Geo)
    Geo.remove_tetrahedra(tets4_cells)
    tets4_cells = np.unique(np.sort(tets4_cells, axis=1), axis=0)
    ghost_nodes_without_debris = np.setdiff1d(Geo.XgID, Geo.RemovedDebrisCells)

    Xs = np.unique(old_tets)
    Xs_c = Xs[~np.isin(Xs, ghost_nodes_without_debris)]
    intercalation_flip = 0
    if len(Xs_c) == 4:
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
    max_pairs_of_edges = old_tets.shape[0] - 3
    # Get all the unique pair of nodes (edges) that can be removed from the mesh
    possibleEdgesToRemove = list(combinations(possibleEdges, max_pairs_of_edges))

    # Step 5: For each combination of edges to remove, check if it is valid
    treeOfPossibilities = nx.DiGraph()
    treeOfPossibilities.add_node(2)
    TRemoved = [None, None]
    Tnew = [None, None]
    Ynew = [None, None]
    parentNode = 0
    arrayPos = 2
    endNode = 1
    _, Tnew, TRemoved, treeOfPossibilities, _ = YFlipNM_recursive(old_tets, TRemoved, Tnew, Ynew, old_ys, Geo,
                                                                  possibleEdges,
                                                                  xs_to_disconnect, treeOfPossibilities, parentNode,
                                                                  arrayPos)

    paths = list(nx.all_simple_paths(treeOfPossibilities, parentNode, endNode))
    new_tets_tree = []
    vol_diff = []
    cell_winning = []
    for c_path in paths:
        c_path = np.array(c_path)
        new_tets = np.vstack(old_tets)

        for posPath in c_path[c_path > 1]:
            toAdd = Tnew[posPath]
            toRemove = TRemoved[posPath]
            new_tets = new_tets[~ismember_rows(np.sort(new_tets, 1), np.sort(toRemove, 1))[0]]
            new_tets = np.vstack((new_tets, toAdd))

        it_was_found = False
        for new_tet_tree in new_tets_tree:
            if np.array_equal(np.sort(new_tet_tree, axis=1), np.sort(new_tets, axis=1)):
                it_was_found = True
                break

        if not it_was_found:
            volumes = []
            for tet in new_tets:
                vol = compute_tet_volume(tet, Geo)
                volumes.append(vol)

            volumes = np.array(volumes)
            norm_vols = volumes / np.max(volumes)
            new_tets = np.array(new_tets)
            new_tets = new_tets[norm_vols > 0.05]
            new_vol = np.sum(volumes[norm_vols > 0.05])

            old_vol = 0
            for tet in old_tets:
                vol = compute_tet_volume(tet, Geo)
                old_vol += vol

            if abs(new_vol - old_vol) / old_vol <= 0.005:
                # try:
                if intercalation_flip:
                    Xs_c = Xs[~np.isin(Xs, ghost_nodes_without_debris)]
                    new_tets = np.append(new_tets, [Xs_c], axis=0)

                Geo_new = Geo.copy()
                Geo_new.remove_tetrahedra(old_tets)
                Geo_new.add_tetrahedra(Geo, np.concatenate((new_tets, tets4_cells)), None, Set)
                Geo_new.rebuild(Geo_new.copy(), Set)
                new_tets_tree.append(new_tets)
                vol_diff.append(abs(new_vol - old_vol) / old_vol)
                cell_winning.append(np.sum(np.isin(new_tets, cell_to_intercalate_with)) / len(new_tets))
                # except Exception as ex:
                #     logger.warning(f"Exception on flip remodelling: {ex}")
                #     pass  # handle exception here if necessary

    if len(new_tets_tree) > 0:
        # min_index = vol_diff.index(min(vol_diff))
        max_index = cell_winning.index(max(cell_winning))
        Tnew = new_tets_tree[max_index]
    else:
        Tnew = []

    Geo.add_tetrahedra(Geo, tets4_cells, None, Set)

    Ynew = []

    return Tnew, Ynew


def add_new_info(Tnew_23, Ynew_23, final_tets, final_ys, new_tets, new_ys, removed_tets, removed_ys, tetIds):
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
