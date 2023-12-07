import numpy as np
from scipy.spatial import Delaunay
from itertools import combinations


def removeTetrahedra(Geo, tets4Cells):
    pass


def get4FoldTets(Geo):
    pass


def YFlipNM_recursive(oldTets, TRemoved, Tnew, Ynew, oldYs, Geo, possibleEdges, XsToDisconnect, treeOfPossibilities,
                      parentNode, arrayPos):
    pass


def ComputeTetVolume(tet, Geo):
    pass

def YFlipNM(oldTets, cellToIntercalateWith, oldYs, XsToDisconnect, Geo, Set):
    Xs_gToDisconnect = XsToDisconnect[np.isin(XsToDisconnect, Geo.XgID)]

    # Temporary remove 4-cell tetrahedra
    tets4Cells = get4FoldTets(Geo)
    Geo = removeTetrahedra(Geo, tets4Cells)
    tets4Cells = np.unique(np.sort(tets4Cells, axis=1), axis=0)
    ghostNodesWithoutDebris = np.setdiff1d(Geo.XgID, Geo.RemovedDebrisCells)

    Xs = np.unique(oldTets)
    Xs_c = Xs[~np.isin(Xs, ghostNodesWithoutDebris)]
    intercalationFlip = 0
    if len(Xs_c) == 4:
        intercalationFlip = 1

    # Step 1: Keep the boundary of tets not changed.
    boundaryNodes = np.unique(oldTets)

    # Step 2: Get edges that can be added when removing the other one
    possibleEdges = np.array(list(combinations(boundaryNodes, 2)))

    # Step 3: Select the edge to add
    edgeToConnect = [cellToIntercalateWith, Xs_gToDisconnect]
    possibleEdges = possibleEdges[~np.isin(possibleEdges, edgeToConnect, axis=1)]

    # Step 4: Propagate the change to get the remaining tets
    # Create tetrahedra

    # Remove edge using TetGen's algorithm
    treeOfPossibilities = digraph()
    treeOfPossibilities.add_node(2)
    TRemoved = []
    Tnew = []
    Ynew = []
    parentNode = 1
    arrayPos = 3
    endNode = 2
    _, Tnew, TRemoved, treeOfPossibilities = YFlipNM_recursive(oldTets, TRemoved, Tnew, Ynew, oldYs, Geo, possibleEdges, XsToDisconnect, treeOfPossibilities, parentNode, arrayPos)

    paths = allpaths(treeOfPossibilities, parentNode, endNode)
    newTets_tree = []
    volDiff = []
    cellWinning = []
    for path in paths:
        cPath = path[0]
        newTets = np.vstack(oldTets)

        for posPath in cPath[cPath > 2]:
            toAdd = Tnew[posPath]
            toRemove = TRemoved[posPath]
            newTets = newTets[~np.all(np.sort(newTets, axis=1) == np.sort(toRemove, axis=1), axis=1)]
            newTets = np.vstack((newTets, toAdd))

            it_was_found = False
            for new_tet_tree in new_tets_tree:
                if np.array_equal(np.sort(new_tet_tree, axis=1), np.sort(new_tets, axis=1)):
                    it_was_found = True
                    break

            if not it_was_found:
                volumes = []
                for tet in new_tets:
                    vol = ComputeTetVolume(tet, Geo)
                    volumes.append(vol)

                norm_vols = volumes / np.max(volumes)
                new_tets = new_tets[norm_vols > 0.05]
                new_vol = np.sum(volumes[norm_vols > 0.05])

                old_vol = 0
                for tet in old_tets:
                    vol = ComputeTetVolume(tet, Geo)
                    old_vol += vol

                if abs(new_vol - old_vol) / old_vol <= 0.005:
                    try:
                        if intercalation_flip:
                            Xs_c = Xs[~np.isin(Xs, ghost_nodes_without_debris)]
                            new_tets = np.append(new_tets, [Xs_c], axis=0)
                        Geo_new = RemoveTetrahedra(Geo, old_tets)
                        Set['TryingFlips'] = 1
                        Geo_new = AddTetrahedra(Geo_new, Geo, np.concatenate((new_tets, tets4_cells)), [], Set)
                        Set['TryingFlips'] = 0
                        Rebuild(Geo_new, Set)
                        new_tets_tree.append(new_tets)
                        vol_diff.append(abs(new_vol - old_vol) / old_vol)
                        cell_winning.append(np.sum(np.isin(new_tets, cell_to_intercalate_with)) / len(new_tets))
                    except Exception as ex:
                        pass  # handle exception here if necessary

