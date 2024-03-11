import numpy as np

from src.pyVertexModel.util.utils import ismember_rows, copy_non_mutable_attributes


class DegreesOfFreedom:
    def __init__(self, mat_file=None):
        """
        Initialize the degrees of freedom.
        :param mat_file:
        """
        self.remodel = None
        if mat_file is None:
            self.Free = []
            self.Fix = []
            self.FixP = []
            self.FixC = []
        else:
            self.Free = np.array(mat_file['Free'][0][0][:, 0], dtype=int) - 1
            self.Fix = np.array(mat_file['Fix'][0][0][:, 0], dtype=int) - 1
            self.FixP = np.array(mat_file['FixP'][0][0][:, 0], dtype=int) - 1
            self.FixC = np.array(mat_file['FixC'][0][0][:, 0], dtype=int) - 1

    def ApplyBoundaryCondition(self, t, Geo, Set):
        """
        Apply the boundary condition to the degrees of freedom. The boundary condition can be either stretching or
        compressing.
        :param t:
        :param Geo:
        :param Set:
        :return:
        """
        if Set.TStartBC <= t <= Set.TStopBC:
            dimP, FixIDs = np.unravel_index(self.FixP, (Geo.numY + Geo.numF + Geo.nCells, 3))
            if Set.BC == 1:
                self.update_dofs_stretch(FixIDs, Geo, Set)
            elif Set.BC == 2:
                self.update_do_fs_compress(Geo, Set)
            self.Free = [dof for dof in self.Free if dof not in self.FixP]
            self.Free = [dof for dof in self.Free if dof not in self.FixC]

    def get_dofs(self, Geo, Set):
        """
        Get the degrees of freedom. The degrees of freedom are the vertices that are not constrained or prescribed.
        :param Geo:
        :param Set:
        :return:
        """
        # Define free and constrained vertices
        dim = 3
        g_constrained = np.zeros((Geo.numY + Geo.numF + Geo.nCells) * 3, dtype=bool)
        g_prescribed = np.zeros((Geo.numY + Geo.numF + Geo.nCells) * 3, dtype=bool)

        if Geo.BorderCells is not None and len(Geo.BorderCells) > 0:
            borderIds = np.concatenate([Geo.Cells[cell].globalIds for cell in Geo.BorderCells])
        else:
            borderIds = []

        for cell in [cell for cell in Geo.Cells if cell.AliveStatus == 1]:
            Y = cell.Y
            gIDsY = cell.globalIds

            for face in cell.Faces:
                if face.Centre[1] < Set.VFixd:
                    g_constrained[(dim * face.globalIds):(dim * (face.globalIds + 1))] = 1
                elif face.Centre[1] > Set.VPrescribed:
                    g_prescribed[dim * face.globalIds + 1] = 1
                    if Set.BC == 1:
                        g_constrained[dim * face.globalIds] = 1
                        g_constrained[dim * face.globalIds + 2] = 1

            fixY = (Y[:, 1] < Set.VFixd) | np.isin(gIDsY, borderIds)
            preY = (Y[:, 1] > Set.VPrescribed) | np.isin(gIDsY, borderIds)

            for idx in np.where(fixY)[0]:
                g_constrained[(dim * gIDsY[idx]): (dim * (gIDsY[idx] + 1))] = 1

            for idx in np.where(preY)[0]:
                g_prescribed[(dim * gIDsY[idx]): (dim * (gIDsY[idx] + 1))] = 1

        self.Free = np.array(np.where((g_constrained == 0) & (g_prescribed == 0))[0], dtype=int)
        self.Fix = np.array(np.concatenate([np.where(g_constrained)[0], np.where(g_prescribed)[0]]), dtype=int)
        self.FixP = np.array(np.where(g_prescribed)[0], dtype=int)
        self.FixC = np.array(np.where(g_constrained)[0], dtype=int)

    def get_remodel_dofs(self, t_new, geo):
        """
        Get the degrees of freedom for the remodelling.
        :param t_new:
        :param geo:
        :return:
        """
        dim = 3
        self.remodel = np.zeros((geo.numY + geo.numF + geo.nCells) * 3, dtype=bool)

        id_tnew = np.unique(t_new)
        id_tnew_cells = np.setdiff1d(id_tnew, geo.XgID)
        id_tnew_cells = [cell_id for cell_id in id_tnew_cells if geo.Cells[cell_id].AliveStatus == 1]

        for num_cell in id_tnew_cells:
            cell = geo.Cells[num_cell]
            news = np.sum(np.isin(cell.T, geo.XgID), axis=1) > 2
            news[(np.sum(np.isin(cell.T, id_tnew_cells), axis=1) == 2) &
                 (np.sum(np.isin(cell.T, geo.XgID), axis=1) == 2)] = True
            news[np.sum(np.isin(cell.T, id_tnew_cells), axis=1) >= 3] = True

            if np.sum(np.isin(t_new, geo.XgBottom)) > np.sum(np.isin(t_new, geo.XgTop)):
                news[np.any(np.isin(cell.T, geo.XgTop), axis=1)] = False
            else:
                news[np.any(np.isin(cell.T, geo.XgBottom), axis=1)] = False

            for globalId in cell.globalIds[news]:
                self.remodel[(dim * globalId): (dim * (globalId + 1))] = 1

            for face_r in cell.Faces:
                face_tets = cell.T[np.unique([tri.Edge for tri in face_r.Tris])]
                if np.any(ismember_rows(face_tets, cell.T[news])[0]):
                    self.remodel[(dim * face_r.globalIds): (dim * (face_r.globalIds + 1))] = 1

        geo.AssemblegIds = np.array(np.where(self.remodel)[0], dtype=int)

        return geo

    def update_do_fs_compress(self, Geo, Set):
        """
        Update the degrees of freedom for the compressing boundary condition.
        :param Geo:
        :param Set:
        :return:
        """
        maxY = Geo.Cells[0].Y[0, 1]

        for cell in Geo.Cells:
            hit = np.where(cell.Y[:, 1] > maxY)[0]
            if len(hit) > 0:
                maxY = np.max(cell.Y[hit, 1])

            for face in cell.Faces:
                if face.Centre[1] > maxY:
                    maxY = face.Centre[1]

        Set.VPrescribed = maxY - Set.dx / ((Set.TStopBC - Set.TStartBC) / Set.dt)
        self.get_dofs(Geo, Set)

        dimP, numP = np.unravel_index(self.FixP, (3, Geo.numY + Geo.numF + Geo.nCells))

        for cell in Geo.Cells:
            prescYi = np.isin(cell.globalIds, numP)
            cell.Y[prescYi, dimP] = Set.VPrescribed

            for gn in range(len(numP)):
                for face in cell.Faces:
                    if numP[gn] == face.globalIds:
                        face.Centre[dimP[gn]] = Set.VPrescribed

        return Geo

    def update_dofs_stretch(self, FixP, Geo, Set):
        """
        Update the degrees of freedom for the stretching boundary condition.
        :param FixP:
        :param Geo:
        :param Set:
        :return:
        """
        for cell in Geo.Cells:
            prescYi = np.isin(cell.globalIds, FixP)
            cell.Y[prescYi, 1] = cell.Y[prescYi, 1] + Set.dx / ((Set.TStopBC - Set.TStartBC) / Set.dt)

            for gn in range(len(FixP)):
                for face in cell.Faces:
                    if FixP[gn] == face.globalIds:
                        face.Centre[1] = face.Centre[1] + Set.dx / (
                                (Set.TStopBC - Set.TStartBC) / Set.dt)

    def copy(self):
        """
        Copy the degrees of freedom.
        :return:
        """
        dof_copy = DegreesOfFreedom()

        copy_non_mutable_attributes(self, '', dof_copy)

        return dof_copy
