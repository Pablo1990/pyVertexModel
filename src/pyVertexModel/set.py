import math
from datetime import datetime

import numpy as np


class Set:
    def __init__(self):
        ## =============================  Topology ============================
        self.SeedingMethod = 1
        self.s = 1.5
        self.ObtainX = 0
        ## Type of inputto obtain  the initial topology of the cells
        self.InputGeo = 'Bubbles'
        self.CellHeight = 15
        self.TotalCells = 40
        ## ===========================  Add Substrate =========================
        self.Substrate = False
        self.kSubstrate = 0
        ## ============================ Time ==================================
        self.tend = 61
        ## ============================ Mechanics =============================
        # Volumes
        self.lambdaV = 5
        self.lambdaV_Debris = 0.001
        # Surface area
        self.SurfaceType = 1
        self.A0eq0 = True
        self.lambdaS1 = 0.5
        self.lambdaS1CellFactor = []
        self.lambdaS2CellFactor = []
        self.lambdaS3CellFactor = []
        self.lambdaS4CellFactor = []
        # Tri energy Area
        self.EnergyBarrierA = True
        self.lambdaB = 5
        self.Beta = 1
        # Tri energy Aspect ratio
        self.EnergyBarrierAR = True
        self.lambdaR = 5
        # Bending
        self.Bending = False
        self.lambdaBend = 0.01
        self.BendingAreaDependent = True
        # Propulsion
        self.Propulsion = False
        # Confinment
        self.Confinement = False
        # Contractility
        self.Contractility = True
        self.cLineTension = 0.0001
        self.noiseContractility = 0.1
        # In plane elasticity
        self.InPlaneElasticity = True
        self.mu_bulk = 3000
        self.lambda_bulk = 2000
        # Substrate
        self.Substrate = 2
        self.kSubstrate = 500
        ## ============================ Viscosity =============================
        self.nu = 1000
        self.LocalViscosityEdgeBased = False
        self.nu_Local_EdgeBased = 0
        self.LocalViscosityOption = 2
        ## =========================== Remodelling ============================
        self.Remodelling = True
        self.RemodelTol = 0
        self.contributionOldYs = 0
        self.RemodelStiffness = 0.6
        self.Reset_PercentageGeo0 = 0.15
        ## ============================ Solution ==============================
        self.tol = 1e-08
        self.MaxIter = 50
        self.Parallel = False
        self.Sparse = False
        self.lastTConverged = 0
        ## ================= Boundary Condition and loading selfting ===========
        self.BC = None
        self.VFixd = -math.inf
        self.VPrescribed = math.inf
        self.dx = 2
        self.TStartBC = 20
        self.TStopBC = 200
        ## =========================== PostProcessing =========================
        self.diary = False
        self.OutputRemove = True
        self.VTK = True
        self.gVTK = False
        self.VTK_iter = False
        self.SaveWorkspace = False
        self.SaveSetting = False
        self.log = 'log.txt'
        ## ========================= Derived variables ========================
        self.Nincr = self.tend * 2
        ## ========================= Derived variables ========================
        self.RemodelingFrequency = (self.tend / self.Nincr)
        self.lambdaS2 = self.lambdaS1 * 0.1
        self.lambdaS3 = self.lambdaS1 / 10
        self.lambdaS4 = self.lambdaS1 / 10
        self.SubstrateZ = - self.CellHeight / 2
        self.f = self.s / 2
        self.nu_LP_Initial = self.nu

        self.BarrierTri0 = 0.001 * self.s

        self.nu0 = self.nu
        self.dt0 = self.tend / self.Nincr
        self.dt = self.dt0
        self.MaxIter0 = self.MaxIter
        self.contributionOldFaceCentre = self.contributionOldYs

        ## Wound variables
        self.woundDefault()

        current_datetime = datetime.now()
        self.OutputFolder = ['Result/', str(current_datetime.strftime("%m-%d_%H%M%S_")), self.InputGeo, '_Cells_',
                             str(self.TotalCells), '_visc_', str(self.nu), '_lVol_', str(self.lambdaV),
                             '_muBulk_', str(self.mu_bulk), '_lBulk_', str(self.lambda_bulk), '_kSubs_',
                             str(self.kSubstrate), '_lt_', str(self.cLineTension), '_noise_',
                             str(self.noiseContractility), '_pString_', str(self.purseStringStrength),
                             '_eTriAreaBarrier_', str(self.lambdaB), '_eARBarrier_', str(self.lambdaR),
                             '_RemStiff_', str(self.RemodelStiffness), '_lS1_', str(self.lambdaS1), '_lS2_',
                             str(self.lambdaS2), '_lS3_', str(self.lambdaS3)]


    def NoBulk_110(self):
        self.InputGeo = 'VertexModelTime'
        # 40 cells; 3 cells to ablate
        # 110 cells; 7 cells to ablate
        self.TotalCells = 110

        self.InPlaneElasticity = False
        self.mu_bulk = 0

        self.lambda_bulk = 0

        self.nu = 5000
        self.Nincr = 61
        self.lambdaB = 1

        self.lambdaR = 0.3

        self.lambdaV = 10
        self.kSubstrate = 100
        self.cLineTension = 0.05
        self.purseStringStrength = 12

        self.noiseContractility = 0
        self.DelayedAdditionalContractility = 0
        # Soft < 0
        # Stiff > 0
        self.RemodelStiffness = 0
        self.lambdaS1 = 10
        self.lambdaS2 = self.lambdaS1 / 10
        self.lambdaS3 = self.lambdaS1 / 10
        self.lambdaS4 = self.lambdaS3
        self.VTK = True

    def woundDefault(self):
        ## ============================== Ablation ============================
        self.Ablation = True
        self.TInitAblation = 1
        self.TEndAblation = self.tend
        self.lambdaSFactor_Debris = np.finfo(float).eps
        ## =========================== Contractility ==========================
        self.Contractility = 0
        self.DelayedAdditionalContractility = 0
        self.purseStringStrength = 10
        self.Contractility_Variability_PurseString = np.multiply(np.array(
            [1, 0.96, 1.087, 1.74, 2.37, 2.61, 2.487, 2.536, 2.46, 2.52, 2.606, 2.456, 2.387, 2.52, 2.31, 2.328,
             2.134, 2.07, 2.055, 1.9, 1.9]), self.purseStringStrength)
        self.Contractility_Variability_LateralCables = np.array(
            [0.45, 0.53, 0.76, 1.15, 1.28, 1.22, 1.38, 1.33, 1.28, 1.4, 1.25, 1.298, 1.45, 1.31, 1.29, 1.42, 1.31,
             1.41, 1.42, 1.37, 1.28])
        self.Contractility_TimeVariability = (np.arange(0, 60 + 3, 3)) / 60 * (self.TEndAblation - self.TInitAblation)

    def menu_input(self, inputMode=None, batchMode=None):
        if inputMode == 7:
            self.NoBulk_110()


    def UpdateSet_F(self, Geo=None):
        self.f_Init = 0.75
        self.f = self.f_Init * np.mean((np.array([Geo.Cells.Vol])) / (np.array([Geo.Cells.Vol0]))) ** 3