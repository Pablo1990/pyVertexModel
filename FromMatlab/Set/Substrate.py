import numpy as np
Set.InputGeo = 'Bubbles'
Geo.nx = 3
Geo.ny = 1
Geo.nz = 1
Set.TotalCells = 40
Set.Ablation = 0
Geo.cellsToAblate = np.array([5])
Set.lambdaV = 20
Set.tend = 200
Set.Nincr = 400
Set.Substrate = 2
Set.kSubstrate = 0.01
Set.SubstrateZ = - 0.9
Set.Contractility = True
Set.cLineTension = 0.001
Set.Remodelling = 0
Set.RemodelingFrequency = 0.5
Set.lambdaB = 0.0001

Set.BC = 2
Set.dx = 1

Set.VPrescribed = realmax
Set.VFixd = - 1
Set.lambdaS1 = 1
Set.lambdaS2 = 1

Set.ApplyBC = True
Set.OutputFolder = 'Result/Remodelling'