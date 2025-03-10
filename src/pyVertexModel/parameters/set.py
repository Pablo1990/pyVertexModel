import logging
import math
import os
from datetime import datetime

import numpy as np
import scipy

from src import PROJECT_DIRECTORY
from src.pyVertexModel.util.utils import copy_non_mutable_attributes

logger = logging.getLogger("pyVertexModel")

class Set:
    def __init__(self, mat_file=None):
        self.percentage_scutoids = 0
        self.myosin_pool = None
        self.deform_array_Z = None
        self.edge_length_threshold = 0.3
        self.kCeiling = None
        self.Contractility_external_axis = None
        self.export_images = True
        self.cLineTension_external = None
        self.Contractility_external = False
        self.initial_filename_state = 'Input/wing_disc_150.mat'
        self.delay_lateral_cables = 5.8
        self.delay_purse_string = self.delay_lateral_cables
        self.ref_A0 = None
        self.lateralCablesStrength = 0
        self.tol0 = None
        self.dt = None
        self.implicit_method = False
        self.TypeOfPurseString = None
        self.Contractility_TimeVariability = None
        self.Contractility_Variability_LateralCables = None
        self.Contractility_Variability_PurseString = None
        self.purseStringStrength = 0
        self.RemodelingFrequency = None
        self.i_incr = None
        self.iter = None
        self.ablation = True
        self.lumen_V0 = None
        self.cell_V0 = None
        self.cell_A0 = None
        self.lumen_axis1 = None
        self.lumen_axis2 = None
        self.lumen_axis3 = None
        self.ellipsoid_axis1 = None
        self.ellipsoid_axis2 = None
        self.ellipsoid_axis3 = None
        self.nu_bottom = None
        # ============================== Ablation ============================
        self.cellsToAblate = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.TInitAblation = 20
        self.TEndAblation = self.TInitAblation + 60
        self.debris_contribution = np.finfo(float).eps
        if mat_file is None:
            # =============================  Topology ============================
            self.SeedingMethod = 1
            self.s = 1.5
            self.ObtainX = 0
            # Type of input to obtain  the initial topology of the cells
            self.InputGeo = 'VertexModelTime'
            self.CellHeight = 15
            self.TotalCells = 150
            # ===========================  Add Substrate =========================
            self.Substrate = 2
            self.kSubstrate = 0
            # ============================ Time ==================================
            self.tend = 60+20
            self.Nincr = self.tend * 100
            # ============================ Mechanics =============================
            # Volumes
            self.lambdaV = 1e-1
            self.lambdaV_Debris = 1e-8
            self.ref_V0 = 1.0
            # Surface area
            self.SurfaceType = 1
            self.A0eq0 = True
            # Top
            self.lambdaS1 = 1.4
            # c_cell-c_cell
            self.lambdaS2 = self.lambdaS1 / 10
            # Bottom
            self.lambdaS3 = self.lambdaS1 / 100
            # Substrate - c_cell
            self.lambdaS4 = self.lambdaS2
            self.ref_A0 = 0.92
            # Tri energy Area
            self.EnergyBarrierA = False
            self.lambdaB = 3.0
            self.Beta = 1.0
            # Tri energy Aspect ratio
            self.EnergyBarrierAR = True
            self.lambdaR = 8e-7
            # Bending
            self.Bending = False
            self.lambdaBend = 0.01
            self.BendingAreaDependent = True
            # Propulsion
            self.Propulsion = False
            # Confinement
            self.Confinement = False
            # Contractility
            self.Contractility = False
            self.cLineTension = 0.0001
            self.noise_random = 0
            # In plane elasticity
            self.InPlaneElasticity = False
            self.mu_bulk = 3000
            self.lambda_bulk = 2000
            # Brownian motion
            self.brownian_motion = False
            self.brownian_motion_scale = 0
            # ============================ Viscosity =============================
            self.nu = 0.07
            self.LocalViscosityEdgeBased = False
            self.nu_Local_EdgeBased = 0
            self.LocalViscosityOption = 2
            # =========================== remodelling ============================
            self.Remodelling = True
            self.contributionOldYs = 0
            self.RemodelStiffness = 0.7
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
            self.VTK = False
            self.gVTK = False
            self.VTK_iter = False
            self.SaveWorkspace = False
            self.SaveSetting = False
        else:
            self.read_mat_file(mat_file)

    def check_for_non_used_parameters(self):
        """
        Check for non-used parameters and put the alternative to zero
        :return:
        """
        if not self.EnergyBarrierA:
            self.lambdaB = 0

        if not self.EnergyBarrierAR:
            self.lambdaR = 0

        if not self.Bending:
            self.lambdaBend = 0

        if not self.Contractility_external:
            self.cLineTension_external = 0

        if not self.Contractility:
            self.cLineTension = 0

        if not self.brownian_motion:
            self.brownian_motion_scale = 0

        if self.implicit_method is False:
            self.tol = self.nu
            self.tol0 = self.nu/20

        if self.Remodelling:
            self.RemodelStiffness = 0.7
        else:
            self.RemodelStiffness = 2

    def redirect_output(self):
        os.makedirs(self.OutputFolder, exist_ok=True)
        os.makedirs(self.OutputFolder + '/images', exist_ok=True)
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
        Define a parameter if it is not defined or is None
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
        self.define_if_not_defined("Nincr", self.tend * 100)
        self.define_if_not_defined("dt", self.tend / self.Nincr)
        self.define_if_not_defined("RemodelingFrequency", 0.1)
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
        self.define_if_not_defined("nu_bottom", self.nu * 600)

        current_datetime = datetime.now()
        new_outputFolder = ''.join([PROJECT_DIRECTORY, '/Result/', str(current_datetime.strftime("%m-%d_%H%M%S_")),
                            'noise_', '{:0.2e}'.format(self.noise_random), '_bNoise_', '{:0.2e}'.format(self.brownian_motion_scale),
                            '_lVol_', '{:0.2e}'.format(self.lambdaV), '_refV0_', '{:0.2e}'.format(self.ref_V0),
                            '_kSubs_', '{:0.2e}'.format(self.kSubstrate),
                            '_lt_', '{:0.2e}'.format(self.cLineTension),
                            '_refA0_', '{:0.2e}'.format(self.ref_A0),
                            '_eARBarrier_', '{:0.2e}'.format(self.lambdaR),
                            '_RemStiff_', str(self.RemodelStiffness), '_lS1_', '{:0.2e}'.format(self.lambdaS1),
                            '_lS2_', '{:0.2e}'.format(self.lambdaS2), '_lS3_', '{:0.2e}'.format(self.lambdaS3),
                            '_ps_', '{:0.2e}'.format(self.purseStringStrength), '_lc_', '{:0.2e}'.format(self.lateralCablesStrength)])
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
        self.RemodelStiffness = 0.1

    def wing_disc_apical_constriction(self):
        self.nu_bottom = self.nu

        self.lambdaR = 8e-10
        self.kSubstrate = 0

        self.ref_A0 = 0.5
        # Top
        self.lambdaS1 = 1.4  # * 0.1
        # c_cell-c_cell
        self.lambdaS2 = self.lambdaS1 / 100
        # Bottom
        self.lambdaS3 = self.lambdaS1 / 1000

        self.ablation = False

        self.check_for_non_used_parameters()

    def wing_disc(self):
        #self.initial_filename_state = 'Input/Stack.tif'
        self.percentage_scutoids = 0.65

        self.EnergyBarrierA = False
        # Energy Barrier Aspect Ratio
        self.EnergyBarrierAR = True
        if self.EnergyBarrierAR:
            self.lambdaR = 8e-7
        else:
            self.lambdaR = 0

        # Substrate
        self.kSubstrate = 0.1

        # Contractility
        self.Contractility = True
        #self.cLineTension = 1e-4

        # Surface Area
        self.ref_A0 = 0.92
        # Top
        self.lambdaS1 = 1.4
        # c_cell-c_cell
        self.lambdaS2 = self.lambdaS1 / 100
        # Bottom
        self.lambdaS3 = self.lambdaS1 / 10
        # Substrate - c_cell
        self.lambdaS4 = self.lambdaS2

        self.VTK = False

        self.check_for_non_used_parameters()

    def squamous_cells(self):
        self.initial_filename_state = 'Input/squamous_cells.pkl'

        # Surface tension
        self.lambda_s_total = (1.4 + 1.4/10 + 1.4/100) * 0.1
        # Top
        self.lambdaS1 = self.lambda_s_total * 2.5/10
        # c_cell-c_cell
        self.lambdaS2 = self.lambda_s_total * 5/10
        # Bottom
        self.lambdaS3 = self.lambda_s_total * 2.5/10

        # Substrate
        self.kSubstrate = 0.1

        self.check_for_non_used_parameters()

    def cuboidal_cells(self):
        self.initial_filename_state = 'Input/cuboidal_cells.pkl'

        # Surface tension
        self.lambda_s_total = (1.4 + 1.4/10 + 1.4/100) * 0.1
        # Top
        self.lambdaS1 = self.lambda_s_total * 2.5/10
        # c_cell-c_cell
        self.lambdaS2 = self.lambda_s_total * 5/10
        # Bottom
        self.lambdaS3 = self.lambda_s_total * 2.5/10

        # Substrate
        self.kSubstrate = 0.001

        self.check_for_non_used_parameters()

    def columnar_cells(self):
        self.initial_filename_state = 'Input/columnar_cells.pkl'

        # Surface tension
        self.lambda_s_total = (1.4 + 1.4/10 + 1.4/100)
        # Top
        self.lambdaS1 = self.lambda_s_total * 75/100
        # c_cell-c_cell
        self.lambdaS2 = self.lambda_s_total * 5/100
        # Bottom
        self.lambdaS3 = self.lambda_s_total * 20/100

    def wound_default(self):
        # =========================== Contractility ==========================
        self.Contractility = True
        self.TypeOfPurseString = 0
        # 0: Intensity-based purse string
        # 1: Strain-based purse string (delayed)
        # 2: Fixed with linear increase purse string
        self.myosin_pool = 4e-5 + 7e-5
        self.purseStringStrength = 4/11 * self.myosin_pool
        self.lateralCablesStrength = self.myosin_pool - self.purseStringStrength

    def menu_input(self, inputMode=None, batchMode=None):
        if inputMode == 7:
            self.wing_disc()

    def copy(self):
        """
        Copy the degrees of freedom.
        :return:
        """
        set_copy = Set()

        copy_non_mutable_attributes(self, '', set_copy)

        return set_copy
