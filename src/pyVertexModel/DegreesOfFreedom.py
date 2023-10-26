import numpy as np


class DegreesOfFreedom:
    def __int__(self):
        pass
    def ApplyBoundaryCondition(self, t, Geo, Dofs, Set):
        if t >= Set["TStartBC"] and t <= Set["TStopBC"]:
            dimP, FixIDs = np.unravel_index(Dofs["FixP"], (3, Geo["numY"] + Geo["numF"] + Geo["nCells"]))
            if Set["BC"] == 1:
                Geo = self.UpdateDOFsStretch(FixIDs, Geo, Set)
            elif Set["BC"] == 2:
                Geo, Dofs = self.UpdateDOFsCompress(Geo, Set)
            Dofs["Free"] = [dof for dof in Dofs["Free"] if dof not in Dofs["FixP"]]
            Dofs["Free"] = [dof for dof in Dofs["Free"] if dof not in Dofs["FixC"]]
        elif Set["BC"] == 1 or Set["BC"] == 2:
            pass
            # Dofs["Free"] = np.unique(Dofs["Free"])  # Uncomment if Dofs is a numpy array

        return Geo, Dofs

    import numpy as np

    def GetDOFs(self, Geo, Set):
        # Define free and constrained vertices
        dim = 3
        gconstrained = np.zeros((Geo["numY"] + Geo["numF"] + Geo["nCells"]) * 3)
        gprescribed = np.zeros((Geo["numY"] + Geo["numF"] + Geo["nCells"]) * 3)

        borderIds = np.concatenate([cell["globalIds"] for cell in Geo["Cells"][Geo["BorderCells"]]])

        for cell in Geo["Cells"]:
            Y = cell["Y"]
            gIDsY = cell["globalIds"]

            for face in cell["Faces"]:
                if face["Centre"][1] < Set["VFixd"]:
                    gconstrained[dim * (np.array(face["globalIds"]) - 1)] = 1
                elif face["Centre"][1] > Set["VPrescribed"]:
                    gprescribed[dim * (np.array(face["globalIds"]) - 1) + 1] = 1
                    if Set["BC"] == 1:
                        gconstrained[dim * (np.array(face["globalIds"]) - 1)] = 1
                        gconstrained[dim * (np.array(face["globalIds"]) - 1) + 2] = 1

            fixY = (Y[:, 1] < Set["VFixd"]) | np.isin(gIDsY, borderIds)
            preY = (Y[:, 1] > Set["VPrescribed"]) | np.isin(gIDsY, borderIds)

            for idx in np.where(fixY)[0]:
                gconstrained[dim * (gIDsY[idx] - 1): dim * gIDsY[idx]] = 1

            gprescribed[dim * (gIDsY[preY] - 1): dim * gIDsY[preY]] = 1

        Dofs = {}
        Dofs["Free"] = np.where((gconstrained == 0) & (gprescribed == 0))[0]
        Dofs["Fix"] = np.concatenate([np.where(gconstrained)[0], np.where(gprescribed)[0]])
        Dofs["FixP"] = np.where(gprescribed)[0]
        Dofs["FixC"] = np.where(gconstrained)[0]

        return Dofs

    import numpy as np

    def UpdateDOFsCompress(self, Geo, Set):
        maxY = Geo["Cells"][0]["Y"][0, 1]

        for cell in Geo["Cells"]:
            hit = np.where(cell["Y"][:, 1] > maxY)[0]
            if len(hit) > 0:
                maxY = np.max(cell["Y"][hit, 1])

            for face in cell["Faces"]:
                if face["Centre"][1] > maxY:
                    maxY = face["Centre"][1]

        Set["VPrescribed"] = maxY - Set["dx"] / ((Set["TStopBC"] - Set["TStartBC"]) / Set["dt"])
        Dofs = self.GetDOFs(Geo, Set)

        dimP, numP = np.unravel_index(Dofs["FixP"], (3, Geo["numY"] + Geo["numF"] + Geo["nCells"]))

        for cell in Geo["Cells"]:
            prescYi = np.isin(cell["globalIds"], numP)
            cell["Y"][prescYi, dimP] = Set["VPrescribed"]

            for gn in range(len(numP)):
                for face in cell["Faces"]:
                    if numP[gn] == face["globalIds"]:
                        face["Centre"][dimP[gn]] = Set["VPrescribed"]

        return Geo, Dofs

    import numpy as np

    def UpdateDOFsStretch(self, FixP, Geo, Set):
        for cell in Geo["Cells"]:
            prescYi = np.isin(cell["globalIds"], FixP)
            cell["Y"][prescYi, 1] = cell["Y"][prescYi, 1] + Set["dx"] / ((Set["TStopBC"] - Set["TStartBC"]) / Set["dt"])

            for gn in range(len(FixP)):
                for face in cell["Faces"]:
                    if FixP[gn] == face["globalIds"]:
                        face["Centre"][1] = face["Centre"][1] + Set["dx"] / (
                                    (Set["TStopBC"] - Set["TStartBC"]) / Set["dt"])

        return Geo
