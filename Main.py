import numpy as np

from Src.InitializeVertexModel import InitializeVertexModel
from Src.IterateOverTime import IterateOverTime
from Src.Set import Set

Sets = np.array([])
Geos = np.array([])
batchMode = 0
inputMode = 7

set = Set()
Set = set.menu_input(inputMode, batchMode)
Sets[0] = Set
Geos[0] = Geo()
tlines = np.array(['"Single execution"'])


numLine = 0
prevLog = ''
didNotConverge = False
try:
    Geo = Geos[numLine]
    Set = Sets[numLine]
    Set,Geo,Dofs,t,tr,Geo_0,backupVars,Geo_n,numStep,relaxingNu,EnergiesPerTimeStep = InitializeVertexModel(Set,Geo)
    while t <= Set.tend and not didNotConverge :
        Geo,Geo_n,Geo_0,Set,Dofs,EnergiesPerTimeStep,t,numStep,tr,relaxingNu,backupVars,didNotConverge = IterateOverTime(Geo,Geo_n,Geo_0,Set,Dofs,EnergiesPerTimeStep,t,numStep,tr,relaxingNu,backupVars)

finally:
    pass
