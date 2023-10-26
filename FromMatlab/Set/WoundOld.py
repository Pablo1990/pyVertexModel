## geometry
import numpy as np
Geo.InputSegmentedImage = 'InputImage_dWP3.bmp'
Set.CellHeight = 15

Set.zScale = 19.23

Set.EllipseFitDiameter = 1

Set.AvgCellArea = np.pi * (Set.EllipseFitDiameter / 2) ** 2

Set.CellHeight = (Set.CellHeight * Set.zScale) / Set.AvgCellArea
Set.TotalCells = 150

##  Mechanics
#---------- Volume
Set.lambdaV = 30
Set.lambdaV_Debris = eps
#---------- Surface
# Set.SurfaceType=4 : Surface-Energy based on the whole cell area differential adhsion
Set.SurfaceType = 4
# external
Set.lambdaS1 = 0.001
# Cell-Cell
Set.lambdaS2 = 0.001
# Cell-substrate
Set.lambdaS3 = Set.lambdaS1
#---------- In plane elasticity
Set.InPlaneElasticity = 1
Set.mu_bulk = 5000

#Set.mu_bulk = Set.cLineTension*1664;
Set.lambda_bulk = 5000

#--------- Bending
Set.Bending = False
#------- Viscosity
Set.nu = 0.5

# Set.Confinement=true;
# Set.ConfinementX1=0.5;
# Set.ConfinementY1=0.5;
# Set.ConfinementX2=-0.5;
# Set.ConfinementY2=-0.5;

## Compression or stretching
Set.BC = 2

Set.VFixd = - 1.5
Set.VPrescribed = 1.5
Set.dx = 0
Set.TStartBC = 301

Set.TStopBC = 302
## Substrate
Set.Substrate = True
Set.kSubstrate = 100

## time
Set.tend = 0.042

#Set.Nincr=Set.tend*100000;
Set.Nincr = Set.tend * 500000
## Remodeling
Set.Remodelling = False
Set.RemodelTol = 0.005
Set.RemodelingFrequency = Set.tend / Set.Nincr
#---------- EnergyBarrier
Set.EnergyBarrier = True
Set.lambdaB = 1
Set.Beta = 1
Set.BarrierTri0 = 0.05

# WBexp =exp( lambdaB*  ( 1 - Set.Beta*At/At0 )  );

## Ablating cells
Set.Ablation = True
#Set.cellsToAblate = findCentralCells(Example(Set.e), 1);
# Aim: Set.cellsToAblate = 1:15;
Set.cellsToAblate = np.arange(1,15+1)
Set.TInitAblation = 0.0001
ablationDuration = 0.06
Set.TEndAblation = Set.TInitAblation + ablationDuration + Set.tend / Set.Nincr

Set.LateralCablesMultiplier = 2
Set.PurseStringMultiplier = 2
Set.additionalFileNameInfo = 'LateralContractiliyGradient_x2_PurseString_x2'
#---------- Line tension
Set.cLineTension = 1
## Contractility
# 0: No contractility
# 1: Lateral cables end-to-end
# 2: Lateral surface contractility
Set.Contractility = 1
# Set.cPurseString = 14;
# Set.cLateralCables = 0.33;

## Execution parameters
Set.batchProcessing = 1
Set.VTK = 0
Set.diary = True
Set.MaxIter = 200
Set.Parallel = False