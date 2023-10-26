import numpy as np
Set.InputGeo = 'VertexModelTime'
# 40 cells; 3 cells to ablate
# 110 cells; 5 cells to ablate
Set.TotalCells = 40
Geo.cellsToAblate = np.array([np.arange(1,3+1)])
Set.InPlaneElasticity = False
Set.mu_bulk = 0

Set.lambda_bulk = 0

Set.nu = 5000
Set.Nincr = 61 * 2
Set.lambdaB = 1

Set.lambdaR = 3
Set.lambdaV = 10
Set.kSubstrate = 1000
Set.cLineTension = 0.2
Set.cLineTensionMembrane = 0
Set.purseStringStrength = 1.5
Set.lambdaS1 = 1
Set.lambdaS2 = 0.01 * Set.lambdaS1
Set.VTK = True