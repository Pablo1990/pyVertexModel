import copy

import numpy as np


class Remodelling:
    def __init__(self, Geo, Geo_n, Geo_0, Set, Dofs):
        """

        :param Geo:
        :param Geo_n:
        :param Geo_0:
        :param Set:
        :param Dofs:
        """
        self.Geo = Geo
        self.Set = Set
        self.Dofs = Dofs
        self.Geo_n = Geo_n
        self.Geo_0 = Geo_0

    def remodelling(self):
        self.Geo.AssemblegIds = []
        newYgIds = []
        checkedYgIds = []
        segmentFeatures_all = self.GetTrisToRemodelOrdered()
        nonDeadCells = [cell.ID for cell in self.Geo.Cells if cell.AliveStatus is not None]

        while segmentFeatures_all:
            Geo_backup = copy.deepcopy(self.Geo)
            Geo_n_backup = copy.deepcopy(self.Geo_n)
            Geo_0_backup = copy.deepcopy(self.Geo_0)
            Dofs_backup = copy.deepcopy(self.Dofs)

            segmentFeatures = segmentFeatures_all[0]
            segmentFeatures = segmentFeatures[np.unique(segmentFeatures[:, :2], axis=0, return_index=True)[1]]
            segmentFeatures = segmentFeatures[segmentFeatures[:, 5].argsort()]
            self.Set.NeedToConverge = 0
            allTnew = []
            numPair = 0

            cellNode = segmentFeatures[numPair, 0]
            ghostNode = segmentFeatures[numPair, 1]
            cellToIntercalateWith = segmentFeatures[numPair, 2]
            cellToSplitFrom = segmentFeatures[numPair, 3]

            hasConverged = [1]

            while hasConverged[numPair] == 1:
                hasConverged[numPair] = 0

                nodesPair = [cellNode, ghostNode]

                valenceSegment, oldTets, oldYs = edgeValence(self.Geo, nodesPair)

                if sum(np.isin(nonDeadCells, oldTets.flatten())) > 2:
                    Geo_0, Geo_n, Geo, Dofs, Set, newYgIds, hasConverged[numPair], Tnew = FlipNM(nodesPair,
                                                                                                 cellToIntercalateWith,
                                                                                                 oldTets, oldYs,
                                                                                                 Geo_0, Geo_n, Geo,
                                                                                                 Dofs, Set,
                                                                                                 newYgIds)

                    allTnew = np.vstack((allTnew, Tnew))

                sharedNodesStill = getNodeNeighboursPerDomain(Geo, cellNode, ghostNode, cellToSplitFrom)

                if any(np.isin(sharedNodesStill, Geo.XgID)):
                    sharedNodesStill_g = sharedNodesStill[np.isin(sharedNodesStill, Geo.XgID)]
                    ghostNode = sharedNodesStill_g[0]
                else:
                    break

            if hasConverged[numPair]:
                PostProcessingVTK(Geo, Geo_0, Set, Set.iIncr + 1)
                gNodeNeighbours = [getNodeNeighbours(Geo, segmentFeatures[numRow, 1]) for numRow in
                                   range(segmentFeatures.shape[0])]
                gNodes_NeighboursShared = np.unique(np.concatenate(gNodeNeighbours))
                cellNodesShared = gNodes_NeighboursShared[~np.isin(gNodes_NeighboursShared, Geo.XgID)]
                numClose = 0.5
                Geo, Geo_n = moveVerticesCloserToRefPoint(Geo, Geo_n, numClose, cellNodesShared, cellToSplitFrom,
                                                          ghostNode, Tnew, Set)
                PostProcessingVTK(Geo, Geo_0, Set, Set.iIncr + 1)

                Dofs = GetDOFs(Geo, Set)
                Dofs, Geo = GetRemodelDOFs(allTnew, Dofs, Geo)
                Geo, Set, DidNotConverge = SolveRemodelingStep(Geo_0, Geo_n, Geo, Dofs, Set)
                if DidNotConverge:
                    Geo_backup.log = Geo.log
                    Geo = Geo_backup
                    Geo_n = Geo_n_backup
                    Dofs = Dofs_backup
                    Geo_0 = Geo_0_backup
                    Geo.log = f'{Geo.log} =>> Full-Flip rejected: did not converge1\n'
                else:
                    newYgIds = np.unique(np.concatenate((newYgIds, Geo.AssemblegIds)))
                    Geo = UpdateMeasures(Geo)
                    hasConverged = 1
            else:
                # Go back to initial state
                Geo_backup.log = Geo.log
                Geo = copy.deepcopy(Geo_backup)
                Geo_n = copy.deepcopy(Geo_n_backup)
                Dofs = copy.deepcopy(Dofs_backup)
                Geo_0 = copy.deepcopy(Geo_0_backup)
                Geo.log = f'{Geo.log} =>> Full-Flip rejected: did not converge2\n'
            PostProcessingVTK(Geo, Geo_0, Set, Set.iIncr + 1)

            checkedYgIds.extend([[feature[0], feature[1]] for feature in segmentFeatures])

            rowsToRemove = []
            if segmentFeatures_all:
                for numRow in range(len(segmentFeatures_all)):
                    cSegFea = segmentFeatures_all[numRow]
                    if all([feature in checkedYgIds for feature in cSegFea[:2]]):
                        rowsToRemove.append(numRow)
            segmentFeatures_all = [feature for i, feature in enumerate(segmentFeatures_all) if i not in rowsToRemove]

    def GetTrisToRemodelOrdered(self):
        """
        Obtain the edges that are going to be remodeled.
        :return:
        """

        for cell in self.Geo.Cells:
            edges_length_top = np.zeros((len(self.Geo.Cells), 1))
            edges_length_bottom = np.zeros((len(self.Geo.Cells), 1))
            for face in cell.Faces:
                if face.InterfaceType == 'CellCell':
                    for tri in face.Tris:
                        # We want the edges shared by more than one cell
                        if len(tri.SharedByCells) > 1:
                            # Go through the shared cells that are not the current cell
                            for cell_id in tri.SharedByCells:
                                if cell_id != cell.ID:
                                    # We only want edges from the interface top and bottom
                                    if cell.Faces[face.ID].InterfaceType == 'Top':
                                        # Calculate the length of the edge, add its value and save it in a dictionary
                                        edges_length_top[cell.ID] = edges_length_top[
                                                                        cell.ID] + tri.EdgeLength / face.Area
                                    elif cell.Faces[face.ID].InterfaceType == 'Bottom':
                                        # Calculate the length of the edge, add its value and save it in a dictionary
                                        edges_length_bottom[cell.ID] = (
                                                edges_length_bottom[cell.ID] + tri.EdgeLength / face.Area)

            if np.any(edges_length_top > 0):
                avgEdgeLength = np.median(edges_length_top[edges_length_top > 0])
                edges_to_intercalate_Top = (edges_length_top < avgEdgeLength -
                                            (self.Set.RemodelStiffness * avgEdgeLength) & edges_length_top > 0)

                # Obtain the nodes that are going to be remodeled

            if np.any(edges_length_bottom > 0):
                avgEdgeLength = np.median(edges_length_bottom[edges_length_bottom > 0])
                edges_to_intercalate_Bottom = (edges_length_bottom < avgEdgeLength -
                                               (self.Set.RemodelStiffness * avgEdgeLength) & edges_length_bottom > 0)

        return list_edges
