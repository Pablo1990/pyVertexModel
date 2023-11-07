import numpy as np


class DegreesOfFreedom:
    def __init__(self):
        self.Free = []
        self.Fix = []
        self.FixP = []
        self.FixC = []

    def ApplyBoundaryCondition(self, t, Geo, Set):
        if Set.TStartBC <= t <= Set.TStopBC:
            dimP, FixIDs = np.unravel_index(self.FixP, (3, Geo.numY + Geo.numF + Geo.nCells))
            if Set.BC == 1:
                Geo = self.UpdateDOFsStretch(FixIDs, Geo, Set)
            elif Set.BC == 2:
                Geo = self.UpdateDOFsCompress(Geo, Set)
            self.Free = [dof for dof in self.Free if dof not in self.FixP]
            self.Free = [dof for dof in self.Free if dof not in self.FixC]
        elif Set.BC == 1 or Set.BC == 2:
            pass
            # self.Free = np.unique(self.Free)  # Uncomment if self is a numpy array

        return Geo

    def GetDOFs(self, Geo, Set):
        # Define free and constrained vertices
        dim = 3
        gconstrained = np.zeros((Geo.numY + Geo.numF + Geo.nCells) * 3)
        gprescribed = np.zeros((Geo.numY + Geo.numF + Geo.nCells) * 3)

        if Geo.BorderCells is not None:
            borderIds = np.concatenate([cell.globalIds for cell in Geo.Cells[Geo.BorderCells]])
        else:
            borderIds = []

        for ci in Geo.non_dead_cells:
            cell = Geo.Cells[ci]
            Y = cell.Y
            gIDsY = cell.globalIds

            for face in cell.Faces:
                if face.Centre[1] < Set.VFixd:
                    gconstrained[dim * (np.array(face.globalIds) - 1)] = 1
                elif face.Centre[1] > Set.VPrescribed:
                    gprescribed[dim * (np.array(face.globalIds) - 1) + 1] = 1
                    if Set.BC == 1:
                        gconstrained[dim * (np.array(face.globalIds) - 1)] = 1
                        gconstrained[dim * (np.array(face.globalIds) - 1) + 2] = 1

            fixY = (Y[:, 1] < Set.VFixd) | np.isin(gIDsY, borderIds)
            preY = (Y[:, 1] > Set.VPrescribed) | np.isin(gIDsY, borderIds)

            for idx in np.where(fixY)[0]:
                gconstrained[(dim * gIDsY[idx]): (dim * (gIDsY[idx] + 1))] = 1

            for idx in np.where(preY)[0]:
                gprescribed[(dim * gIDsY[idx]): (dim * (gIDsY[idx]+1))] = 1

        self.Free = np.array(np.where((gconstrained == 0) & (gprescribed == 0))[0], dtype=int)
        self.Fix = np.array(np.concatenate([np.where(gconstrained)[0], np.where(gprescribed)[0]]), dtype=int)
        self.FixP = np.array(np.where(gprescribed)[0], dtype=int)
        self.FixC = np.array(np.where(gconstrained)[0], dtype=int)

    def UpdateDOFsCompress(self, Geo, Set):
        maxY = Geo.Cells[0].Y[0, 1]

        for cell in Geo.Cells:
            hit = np.where(cell.Y[:, 1] > maxY)[0]
            if len(hit) > 0:
                maxY = np.max(cell.Y[hit, 1])

            for face in cell.Faces:
                if face.Centre[1] > maxY:
                    maxY = face.Centre[1]

        Set.VPrescribed = maxY - Set.dx / ((Set.TStopBC - Set.TStartBC) / Set.dt)
        self.GetDOFs(Geo, Set)

        dimP, numP = np.unravel_index(self.FixP, (3, Geo.numY + Geo.numF + Geo.nCells))

        for cell in Geo.Cells:
            prescYi = np.isin(cell.globalIds, numP)
            cell.Y[prescYi, dimP] = Set.VPrescribed

            for gn in range(len(numP)):
                for face in cell.Faces:
                    if numP[gn] == face.globalIds:
                        face.Centre[dimP[gn]] = Set.VPrescribed

        return Geo

    def UpdateDOFsStretch(self, FixP, Geo, Set):
        for cell in Geo.Cells:
            prescYi = np.isin(cell.globalIds, FixP)
            cell.Y[prescYi, 1] = cell.Y[prescYi, 1] + Set.dx / ((Set.TStopBC - Set.TStartBC) / Set.dt)

            for gn in range(len(FixP)):
                for face in cell.Faces:
                    if FixP[gn] == face.globalIds:
                        face.Centre[1] = face.Centre[1] + Set.dx / (
                                (Set.TStopBC - Set.TStartBC) / Set.dt)

        return Geo
