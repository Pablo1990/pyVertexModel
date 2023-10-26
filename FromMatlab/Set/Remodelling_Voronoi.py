import numpy as np
Set.InputGeo = 'Voronoi'
Set.TotalCells = 40
Geo.cellsToAblate = np.array([np.arange(1,3+1)])
Set.nu = 1000
Set.lambdaV = 100

Set.mu_bulk = 10000

Set.lambda_bulk = 5000

Set.kSubstrate = 500

Set.cLineTension = 0.1

Set.purseStringStrength = 2
Set.RemodelStiffness = 1
Set.lambdaB = 50
Set.lambdaS1 = 1
Set.lambdaS2 = 0.9
Set.lambdaS3 = 1