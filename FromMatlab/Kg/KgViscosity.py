import numpy as np


def KgViscosity(Geo_n=None, Geo=None, Set=None):
    K = np.multiply((Set.nu / Set.dt), np.eye((Geo.numY + Geo.numF + Geo.nCells) * 3))
    # TODO FIXME placeholder...
    dY = np.zeros((Geo.numF + Geo.numY + Geo.nCells, 3))
    # TODO FIXME BAD!
    for c in np.array([Geo.Cells(not cellfun(isempty, np.array([Geo.Cells.AliveStatus]))).ID]).reshape(-1):
        # THERE WAS A HARD TO DEBUG ERROR HERE...
        if Geo.Remodelling:
            if not np.isin(c, Geo.AssembleNodes):
                continue
        Cell = Geo.Cells(c)
        Cell_n = Geo_n.Cells(c)
        dY[Cell.globalIds, :] = (Cell.Y - Cell_n.Y)
        for f in np.arange(1, len(Cell.Faces) + 1).reshape(-1):
            Face = Cell.Faces(f)
            Face_n = Cell_n.Faces(f)
            if not isstring(Face.Centre) and not isstring(Face_n.Centre):
                dY[Face.globalIds, :] = (Face.Centre - Face_n.Centre)
        dY[Cell.cglobalIds, :] = (Cell.X - Cell_n.X)

    g = np.multiply((Set.nu / Set.dt), reshape(np.transpose(dY), (Geo.numF + Geo.numY + Geo.nCells) * 3, 1))
    EnergyF = (1 / 2) * (np.transpose(g)) * g / Set.nu
    return g, K, EnergyF

    return g, K, EnergyF
