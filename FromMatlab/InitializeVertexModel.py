import numpy as np

from Src.DOFs.GetDOFs import GetDOFs
from Src.DOFs.GetDOFsSubstrate import GetDOFsSubstrate
from Src.Geo.InitializeGeometry3DVertex import InitializeGeometry3DVertex
from Src.Geo.InitializeGeometry_3DVoronoi import InitializeGeometry_3DVoronoi
from Src.Geo.InitializeGeometry_VertexModel2DTime import InitializeGeometry_VertexModel2DTime
from Src.PostProcessing.InitiateOutputFolder import InitiateOutputFolder


def InitializeVertexModel(Set = None,Geo = None):

    Set = InitiateOutputFolder(Set)
    if Set.InputGeo=='Bubbles':
        Geo,Set = InitializeGeometry3DVertex(Geo,Set)
    else:
        if Set.InputGeo=='Voronoi':
            Geo,Set = InitializeGeometry_3DVoronoi(Geo,Set)
        else:
            if Set.InputGeo=='VertexModelTime':
                Geo,Set = InitializeGeometry_VertexModel2DTime(Geo,Set)
    
    minZs = np.amin(vertcat(Geo.Cells(np.arange(1,Geo.nCells+1)).Y))
    if minZs(3) > 0:
        Set.SubstrateZ = minZs(3) * 0.99
    else:
        Set.SubstrateZ = minZs(3) * 1.01
    
    # TODO FIXME, this is bad, should be joined somehow
    if Set.Substrate == 1:
        Dofs = GetDOFsSubstrate(Geo,Set)
    else:
        Dofs = GetDOFs(Geo,Set)
    
    Geo.Remodelling = False
    t = 0
    tr = 0
    Geo_0 = Geo
    # Removing info of unused features from Geo
    Geo_0.Cells.Vol = deal([])
    Geo_0.Cells.Vol0 = deal([])
    Geo_0.Cells.Area = deal([])
    Geo_0.Cells.Area0 = deal([])
    Geo_n = Geo
    Geo_n.Cells.Vol = deal([])
    Geo_n.Cells.Vol0 = deal([])
    Geo_n.Cells.Area = deal([])
    Geo_n.Cells.Area0 = deal([])
    backupVars.Geo_b = Geo
    backupVars.tr_b = tr
    backupVars.Dofs = Dofs
    numStep = 1
    relaxingNu = False
    EnergiesPerTimeStep = np.array([])
    PostProcessingVTK(Geo,Geo_0,Set,numStep)
    return Set,Geo,Dofs,t,tr,Geo_0,backupVars,Geo_n,numStep,relaxingNu,EnergiesPerTimeStep