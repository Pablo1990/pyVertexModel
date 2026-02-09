import logging
import math
import os
from datetime import datetime

import numpy as np
import scipy

from pyVertexModel import PROJECT_DIRECTORY
from pyVertexModel.util.utils import copy_non_mutable_attributes

logger = logging.getLogger("pyVertexModel")

class Set:
    def __init__(self, mat_file=None):
        """
        Initialize simulation configuration attributes with sensible defaults.
        
        When mat_file is None, populate a comprehensive set of simulation parameters (topology, geometry, mechanics, time stepping, substrate, viscosity, remodeling, solver, boundary/loading, postprocessing, and ablation defaults). When mat_file is provided, load parameter values from the given MATLAB-like structure via read_mat_file(mat_file).
        
        Parameters:
            mat_file (optional): A MATLAB-style struct or object containing saved Set parameters; when provided, values are read and assigned to this instance.
        """
        self.dt_tolerance = 1e-6
        self.min_3d_neighbours = None
        self.periodic_boundaries = True
        self.frozen_face_centres = False
        self.frozen_face_centres_border_cells = True
        self.model_name = ''
        self.percentage_scutoids = 0
        self.myosin_pool = 1e-4
        self.resize_z = None
        self.edge_length_threshold = 0.3
        self.kCeiling = None
        self.Contractility_external_axis = None
        self.export_images = True
        self.cLineTension_external = None
        self.Contractility_external = False
        self.initial_filename_state = None
        self.delay_lateral_cables = 5.8
        self.delay_purse_string = self.delay_lateral_cables
        self.ref_A0 = None
        self.lateralCablesStrength = 0
        self.tol0 = None
        self.dt = None
        self.dt0 = None
        self.implicit_method = False
        self.integrator = 'euler'  # Time integrator: 'euler', 'rk2' (midpoint method), or 'fire' (FIRE algorithm)
        
        # FIRE Algorithm parameters (Bitzek et al., 2006)
        # These parameters control the adaptive optimization when integrator='fire'
        self.fire_dt_max = None      # Maximum timestep (will be set to 10*dt if None)
        self.fire_dt_min = None      # Minimum timestep (will be set to 0.02*dt if None)
        self.fire_N_min = 5          # Steps before acceleration (recommended: 5)
        self.fire_f_inc = 1.1        # dt increase factor (recommended: 1.1)
        self.fire_f_dec = 0.5        # dt decrease factor (recommended: 0.5)
        self.fire_alpha_start = 0.1  # Initial damping coefficient (recommended: 0.1)
        self.fire_f_alpha = 0.99     # α decrease factor (recommended: 0.99)

        # Standard FIRE parameters - OPTIMIZED FOR SPEED and WITHOUT TIME DEPENDENCE
        self.fire_alpha_start = 0.2  # More aggressive mixing
        self.fire_f_inc = getattr(self, 'fire_f_inc', 1.25)  # Faster dt increase
        self.fire_f_dec = getattr(self, 'fire_f_dec', 0.2)  # More aggressive dt reduction on failure
        self.fire_f_alpha = getattr(self, 'fire_f_alpha', 0.97)  # Faster α decay
        self.fire_N_min = getattr(self, 'fire_N_min', 2)  # Accelerate sooner
        self.fire_dt_max = getattr(self, 'fire_dt_max', 20.0)  # Large max dt for fast minimization
        self.fire_dt_min = getattr(self, 'fire_dt_min', 1e-8)  # Very small min dt

        # Convergence tolerances - PRACTICAL FOR VERTEX MODELS
        self.fire_force_tol = getattr(self, 'fire_force_tol', 1e-6)  # Tight for steady-state
        self.fire_disp_tol = getattr(self, 'fire_disp_tol', 1e-10)  # Tight displacement
        self.fire_vel_tol = getattr(self, 'fire_vel_tol', 1e-12)  # Tight velocity
        self.fire_max_iterations = getattr(self, 'fire_max_iterations', 500)  # Allow more iterations for tight convergence

        # Additional parameters
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
        self.bottom_ecm = True
        self.substrate_top = False
        self.kSubstrateTop = 0
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
            self.lambdaV = 1
            self.lambdaV_Debris = 1e-8
            self.ref_V0 = 0.9999
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
            self.lambdaS4_top = self.lambdaS2
            self.lambdaS4_bottom = self.lambdaS2
            self.ref_A0 = 0.92
            # Tri energy Area
            self.EnergyBarrierA = False
            self.lambdaB = 5.0
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
            self.cLineTension = 0
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
            self.RemodelStiffness = None
            self.Remodel_stiffness_wound = None
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
        Adjust configuration attributes based on feature toggles, disabling unused options and setting related defaults.
        
        For each boolean toggle on the Set instance, updates dependent parameters as follows:
        - If EnergyBarrierA is False, sets `lambdaB` to 0.
        - If EnergyBarrierAR is False, sets `lambdaR` to 0.
        - If Bending is False, sets `lambdaBend` to 0.
        - If Contractility_external is False, sets `cLineTension_external` to 0.
        - If Contractility is False, sets `cLineTension` to 0.
        - If brownian_motion is False, sets `brownian_motion_scale` to 0.
        - If implicit_method is False, sets `tol` to `nu` and `tol0` to `nu/20`.
        - If Remodelling is True, sets `RemodelStiffness` to 0.9 and `Remodel_stiffness_wound` to 0.7; otherwise sets both to 2.
        
        This method mutates the instance's attributes and does not return a value.
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

        if not self.implicit_method:
            self.tol = self.nu

        if self.Remodelling:
            self.RemodelStiffness = 0.9
            self.Remodel_stiffness_wound = 0.7
        else:
            self.RemodelStiffness = 2
            self.Remodel_stiffness_wound = 2

    def redirect_output(self):
        """
        Configure logging to write to a file under the instance's OutputFolder and ensure the required directories exist.
        
        If self.OutputFolder is not None, this creates the OutputFolder and an images subdirectory, attaches a FileHandler writing to 'log.out' with DEBUG level and a timestamped formatter, and enables logger propagation. If OutputFolder is None, no logging configuration or filesystem changes are made.
        """
        if self.OutputFolder is not None:
            os.makedirs(self.OutputFolder, exist_ok=True)
            os.makedirs(self.OutputFolder + '/images', exist_ok=True)
            handler = logging.FileHandler(os.path.join(self.OutputFolder, 'log.out'))
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.propagate = True

    def read_mat_file(self, mat_file):
        """
        Populate this object's attributes from a MATLAB-like structured array by mapping each top-level field to an attribute of the same name.
        
        Parameters:
            mat_file: array-like
                A MATLAB-style structured array (as returned by scipy.io.loadmat for a struct). For each field in `mat_file`, the corresponding attribute on `self` is set to `None` if the field is empty, otherwise to the field's first nested value.
        """
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
        self.define_if_not_defined("lambdaS4_top", self.lambdaS1 / 10)
        self.define_if_not_defined("lambdaS4_bottom", self.lambdaS1 / 10)
        self.define_if_not_defined("SubstrateZ", - self.CellHeight / 2)
        self.define_if_not_defined("f", self.s / 2)
        self.define_if_not_defined("nu_LP_Initial", self.nu)
        self.define_if_not_defined("BarrierTri0", 0.001 * self.s)
        self.define_if_not_defined("nu0", self.nu)
        self.define_if_not_defined("dt0", self.dt)
        self.define_if_not_defined("MaxIter0", self.MaxIter)
        self.define_if_not_defined("contributionOldFaceCentre", self.contributionOldYs)
        self.define_if_not_defined("nu_bottom", self.nu * 600)

        current_datetime = datetime.now()
        new_outputFolder = ''.join([PROJECT_DIRECTORY, '/Result/', str(current_datetime.strftime("%m-%d_%H%M%S_")),
                                    self.model_name,
                                    '_scutoids_', str(self.percentage_scutoids),
                                    '_noise_', '{:0.2e}'.format(self.noise_random),
                                    '_lVol_', '{:0.2e}'.format(self.lambdaV),
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

    def bubbles(self):
        self.InputGeo = 'Bubbles'
        self.model_name = 'Bubbles_with_substrate'
        self.initial_filename_state = 'Input/Bubbles_with_substrate.mat'
        self.Substrate = 3
        self.periodic_boundaries = False
        self.resize_z = None

        self.lambdaS1 = 1
        self.lambdaS2 = self.lambdaS1 / 100
        self.lambdaS3 = self.lambdaS1
        self.lambdaS4_top = 0.1
        self.lambdaS4_bottom = 0.1

        self.Nincr = self.tend * 100

        self.EnergyBarrierAR = False
        if self.EnergyBarrierAR:
            self.lambdaR = 8e-7
        else:
            self.lambdaR = 0

        # Volume
        self.lambdaV = 1
        self.ref_V0 = 1.1

        # Substrate
        self.kSubstrate = 0.1

        # Surface Area
        self.ref_A0 = 0.99

        self.Remodelling = False

        self.VTK = False

        self.check_for_non_used_parameters()

    def cyst(self):
        mat_info = scipy.io.loadmat(os.path.join(PROJECT_DIRECTORY, 'Tests_data/Geo_var_cyst.mat'))
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
        self.lambdaS1 = 1.4
        # c_cell-c_cell
        self.lambdaS2 = self.lambdaS1 / 100
        # Bottom
        self.lambdaS3 = self.lambdaS1 / 1000

        self.ablation = False

        self.check_for_non_used_parameters()

    def wing_disc_equilibrium(self):
        self.integrator = 'fire'
        self.nu_bottom = self.nu
        self.model_name = 'dWL1'
        self.initial_filename_state = 'Input/images/' + self.model_name + '.tif'
        #self.initial_filename_state = 'Input/images/dWP1_150cells_15_scutoids_1.0.pkl'
        self.percentage_scutoids = 0.0
        self.tend = 20
        self.Nincr = self.tend * 4
        self.CellHeight = 1 * 15 #np.array([0.0001, 0.001, 0.01, 0.1, 0.5, 1, 2.0]) * original_wing_disc_height
        #self.resize_z = 0.01
        self.min_3d_neighbours = None # 10

        self.EnergyBarrierAR = True
        if self.EnergyBarrierAR:
            self.lambdaR = 8e-7
        else:
            self.lambdaR = 0

        # Volume
        self.lambdaV = 1

        # Substrate
        self.kSubstrate = 0.1
        #self.kSubstrateTop = self.kSubstrate / 1000
        self.substrate_top = False

        # Surface Area
        self.ref_A0 = 0.99
        # Top
        self.lambdaS1 = 1.4
        # c_cell-c_cell
        self.lambdaS2 = self.lambdaS1 / 100
        # Bottom
        self.lambdaS3 = self.lambdaS1
        # Substrate - c_cell
        self.lambdaS4_top = self.lambdaS2
        self.lambdaS4_bottom = self.lambdaS2

        self.VTK = False
        self.Remodelling = True
        self.ablation = False
        self.noise_random = 0

        self.check_for_non_used_parameters()

    def wing_disc(self):
        self.frozen_face_centres = False
        #self.nu_bottom = self.nu
        #self.model_name = 'in_silico_movie_0'
        #self.initial_filename_state = 'Input/' + self.model_name + '.mat'
        #self.percentage_scutoids = 0
        self.Nincr = self.tend * 100
        self.resize_z = None

        self.EnergyBarrierA = False
        # Energy Barrier Aspect Ratio
        self.EnergyBarrierAR = True
        if self.EnergyBarrierAR:
            self.lambdaR = 8e-7
        else:
            self.lambdaR = 0

        # Volume
        self.lambdaV = 1

        # Substrate
        self.kSubstrate = 0.1

        # Contractility
        self.Contractility = False
        #self.cLineTension = 0

        # Surface Area
        self.ref_A0 = 0.92
        # Top
        self.lambdaS1 = 1.4
        # c_cell-c_cell
        self.lambdaS2 = self.lambdaS1 / 100
        # Bottom
        self.lambdaS3 = self.lambdaS1
        # Substrate - c_cell/
        self.lambdaS4_top = self.lambdaS2
        self.lambdaS4_bottom = self.lambdaS2

        #self.noise_random = 0.3

        self.Remodel_stiffness_wound = 0.9

        self.VTK = False

        self.check_for_non_used_parameters()

    def wound_default(self, myosin_pool_multiplier=1):
        # =========================== Contractility ==========================
        self.Contractility = True
        self.TypeOfPurseString = 0
        # 0: Intensity-based purse string
        # 1: Strain-based purse string (delayed)
        # 2: Fixed with linear increase purse string
        self.myosin_pool = (3e-5 + 7e-5) * myosin_pool_multiplier
        self.purseStringStrength = 3e-5
        self.lateralCablesStrength = 7e-5

        # fire parameters for faster convergence to equilibrium after ablation
        self.fire_alpha_start = getattr(self, 'fire_alpha_start', 0.15)  # Moderate mixing
        self.fire_f_inc = getattr(self, 'fire_f_inc', 1.15)  # Moderate dt increase
        self.fire_f_dec = getattr(self, 'fire_f_dec', 0.25)  # Quick recovery on bad steps
        self.fire_f_alpha = getattr(self, 'fire_f_alpha', 0.98)  # Moderate α decay
        self.fire_N_min = getattr(self, 'fire_N_min', 2)  # Accelerate quickly
        self.fire_dt_max = getattr(self, 'fire_dt_max', 5.0 * self.dt)  # Limited max dt (prevent overshoot)
        self.fire_dt_min = getattr(self, 'fire_dt_min', 1e-6 * self.dt)  # Very small min dt

        # Convergence tolerances - PRACTICAL FOR DYNAMICAL SIMULATIONS
        self.fire_force_tol = getattr(self, 'fire_force_tol', 5e-3)  # Loose tolerance (0.5% of typical forces)
        self.fire_disp_tol = getattr(self, 'fire_disp_tol', 1e-6)  # Moderate displacement
        self.fire_vel_tol = getattr(self, 'fire_vel_tol', 1e-8)  # Moderate velocity
        self.fire_max_iterations = getattr(self, 'fire_max_iterations', 30)  # STRICT LIMIT for dynamics

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