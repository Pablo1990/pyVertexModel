import logging
import math
import os
import sys
from datetime import datetime

import numpy as np
import scipy

from src import PROJECT_DIRECTORY

logger = logging.getLogger("pyVertexModel")


class Set:
    def __init__(self, mat_file=None):
        self.RemodelingFrequency = None
        self.i_incr = None
        self.iter = None
        self.ablation = False
        self.lumen_V0 = None
        self.cell_V0 = None
        self.cell_A0 = None
        self.lumen_axis1 = None
        self.lumen_axis2 = None
        self.lumen_axis3 = None
        self.ellipsoid_axis1 = None
        self.ellipsoid_axis2 = None
        self.ellipsoid_axis3 = None
        if mat_file is None:
            # =============================  Topology ============================
            self.SeedingMethod = 1
            self.s = 1.5
            self.ObtainX = 0
            # Type of input to obtain  the initial topology of the cells
            self.InputGeo = 'Bubbles'
            self.CellHeight = 15
            self.TotalCells = 40
            # ===========================  Add Substrate =========================
            self.Substrate = True
            self.kSubstrate = 0
            # ============================ Time ==================================
            self.tend = 61
            # ============================ Mechanics =============================
            # Volumes
            self.lambdaV = 5.0
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
            self.lambdaB = 5.0
            self.Beta = 1.0
            # Tri energy Aspect ratio
            self.EnergyBarrierAR = True
            self.lambdaR = 5.0
            # Bending
            self.Bending = False
            self.lambdaBend = 0.01
            self.BendingAreaDependent = True
            # Propulsion
            self.Propulsion = False
            # Confinement
            self.Confinement = False
            # Contractility
            self.Contractility = True
            self.cLineTension = 0.0001
            self.noiseContractility = 0
            # In plane elasticity
            self.InPlaneElasticity = False
            self.mu_bulk = 3000
            self.lambda_bulk = 2000
            # Substrate
            self.Substrate = 2
            self.kSubstrate = 500.0
            # Brownian motion
            self.brownian_motion = False
            self.brownian_motion_scale = 0
            # ============================ Viscosity =============================
            self.nu = 1000.0
            self.LocalViscosityEdgeBased = False
            self.nu_Local_EdgeBased = 0
            self.LocalViscosityOption = 2
            # =========================== Remodelling ============================
            self.Remodelling = False
            self.RemodelTol = 0
            self.contributionOldYs = 0
            self.RemodelStiffness = 0.6
            self.Reset_PercentageGeo0 = 0.15
            # ============================ Solution ==============================
            self.tol = 1e-08
            self.MaxIter = 30
            self.Parallel = False
            self.Sparse = False
            self.last_t_converged = 0
            # ================= Boundary Condition and loading selfting ===========
            self.BC = None
            self.VFixd = -math.inf
            self.VPrescribed = math.inf
            self.dx = 2
            self.TStartBC = 20
            self.TStopBC = 200
            # =========================== PostProcessing =========================
            self.diary = False
            self.OutputRemove = True
            self.VTK = True
            self.gVTK = False
            self.VTK_iter = False
            self.SaveWorkspace = False
            self.SaveSetting = False
        else:
            self.read_mat_file(mat_file)

    def redirect_output(self):
        os.makedirs(self.OutputFolder, exist_ok=True)
        handler = logging.FileHandler(os.path.join(self.OutputFolder, 'log.out'))
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = True

    def read_mat_file(self, mat_file):
        for param in mat_file.dtype.fields:
            if len(mat_file[param][0][0]) == 0:
                setattr(self, param, None)
            else:
                setattr(self, param, mat_file[param][0][0][0][0])

    def define_if_not_defined(self, param, value):
        """
        Define a parameter if it is not defined
        :param param:
        :param value:
        :return:
        """
        if not hasattr(self, param):
            setattr(self, param, value)
        elif self.__dict__[param] is None:
            setattr(self, param, value)

    def update_derived_parameters(self):
        """
        Update derived parameters
        :return:
        """
        self.define_if_not_defined("Nincr", self.tend * 2)
        self.define_if_not_defined("dt", self.tend / self.Nincr)
        self.define_if_not_defined("RemodelingFrequency", self.tend / self.Nincr)
        self.define_if_not_defined("lambdaS2", self.lambdaS1 * 0.1)
        self.define_if_not_defined("lambdaS3", self.lambdaS1 / 10)
        self.define_if_not_defined("lambdaS4", self.lambdaS1 / 10)
        self.define_if_not_defined("SubstrateZ", - self.CellHeight / 2)
        self.define_if_not_defined("f", self.s / 2)
        self.define_if_not_defined("nu_LP_Initial", self.nu)
        self.define_if_not_defined("BarrierTri0", 0.001 * self.s)
        self.define_if_not_defined("nu0", self.nu)
        self.define_if_not_defined("dt0", self.tend / self.Nincr)
        self.define_if_not_defined("MaxIter0", self.MaxIter)
        self.define_if_not_defined("contributionOldFaceCentre", self.contributionOldYs)

        current_datetime = datetime.now()
        new_outputFolder = ''.join(['Result/', str(current_datetime.strftime("%m-%d_%H%M%S_")), self.InputGeo,
                                    '_Cells_', str(self.TotalCells), '_visc_', str(self.nu), '_lVol_',
                                    str(self.lambdaV), '_muBulk_', str(self.mu_bulk), '_lBulk_',
                                    str(self.lambda_bulk), '_kSubs_',
                                    str(self.kSubstrate), '_lt_', str(self.cLineTension), '_noise_',
                                    str(self.noiseContractility),
                                    '_eTriAreaBarrier_', str(self.lambdaB), '_eARBarrier_', str(self.lambdaR),
                                    '_RemStiff_', str(self.RemodelStiffness), '_lS1_', str(self.lambdaS1),
                                    '_lS2_', str(self.lambdaS2), '_lS3_', str(self.lambdaS3)])
        self.define_if_not_defined("OutputFolder", new_outputFolder)

    def stretch(self):
        self.tend = 300
        self.Nincr = 300
        self.BC = 1
        self.dx = 2
        self.lambdaS1 = 1
        self.lambdaS2 = 0.8
        self.VPrescribed = 1.5
        self.VFixd = -1.5
        self.ApplyBC = True
        self.lambdaS3 = 0.1
        self.InputGeo = 'Bubbles'
        self.VTK = False

    def cyst(self):
        mat_info = scipy.io.loadmat(os.path.join(PROJECT_DIRECTORY, 'Tests/data/Geo_var_cyst.mat'))
        self.read_mat_file(mat_info['Set'])
        self.InputGeo = 'Bubbles_Cyst'
        self.CellHeight = 15
        self.OutputFolder = os.path.join(PROJECT_DIRECTORY, 'Result/Cyst')
        self.lambdaR = 0
        self.Remodelling = True

    def NoBulk_110(self):
        self.InputGeo = 'VertexModelTime'
        # 40 cells 3 cells to ablate
        # 110 cells 7 cells to ablate
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
        # ============================== Ablation ============================
        self.ablation = False
        self.TInitAblation = 1
        self.TEndAblation = self.tend
        self.lambdaSFactor_Debris = np.finfo(float).eps
        # =========================== Contractility ==========================
        self.Contractility = True
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