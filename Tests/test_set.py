import unittest

import numpy as np

from pyVertexModel.parameters.set import Set


class TestSet(unittest.TestCase):
    """Tests for the Set parameter class."""

    def test_init_defaults(self):
        """Test that Set initializes with correct default values."""
        s = Set()
        self.assertEqual(s.SeedingMethod, 1)
        self.assertAlmostEqual(s.s, 1.5)
        self.assertEqual(s.TotalCells, 150)
        self.assertEqual(s.CellHeight, 15)
        self.assertTrue(s.periodic_boundaries)
        self.assertFalse(s.frozen_face_centres)
        self.assertEqual(s.tend, 80)
        self.assertEqual(s.lambdaV, 1)
        self.assertFalse(s.Contractility)
        self.assertFalse(s.Bending)
        self.assertFalse(s.ablation is None)

    def test_init_seeding_method(self):
        """Test that Set has a seeding method set."""
        s = Set()
        self.assertIsNotNone(s.SeedingMethod)

    def test_init_surface_area_params(self):
        """Test that Set has surface area parameters."""
        s = Set()
        self.assertIsNotNone(s.lambdaS1)
        self.assertIsNotNone(s.lambdaS2)
        self.assertIsNotNone(s.lambdaS3)

    def test_init_ablation_params(self):
        """Test ablation parameters are initialized."""
        s = Set()
        self.assertIsNotNone(s.cellsToAblate)
        self.assertEqual(s.TInitAblation, 20)

    def test_wing_disc(self):
        """Test wing_disc preset configures parameters correctly."""
        s = Set()
        s.wing_disc()
        self.assertEqual(s.lambdaV, 1)
        self.assertTrue(s.EnergyBarrierAR)
        self.assertFalse(s.frozen_face_centres)
        self.assertEqual(s.kSubstrate, 0.1)
        self.assertAlmostEqual(s.ref_A0, 0.92)

    def test_stretch(self):
        """Test stretch preset configures parameters correctly."""
        s = Set()
        s.stretch()
        self.assertEqual(s.tend, 300)
        self.assertEqual(s.Nincr, 300)
        self.assertEqual(s.BC, 1)
        self.assertEqual(s.dx, 2)
        self.assertEqual(s.InputGeo, 'Bubbles')

    def test_bubbles(self):
        """Test bubbles preset configures parameters correctly."""
        s = Set()
        s.bubbles()
        self.assertEqual(s.InputGeo, 'Bubbles')
        self.assertEqual(s.Substrate, 3)
        self.assertFalse(s.periodic_boundaries)

    def test_wing_disc_apical_constriction(self):
        """Test wing_disc_apical_constriction preset."""
        s = Set()
        s.wing_disc_apical_constriction()
        self.assertEqual(s.kSubstrate, 0)
        self.assertAlmostEqual(s.ref_A0, 0.5)
        self.assertFalse(s.ablation)

    def test_wing_disc_equilibrium(self):
        """Test wing_disc_equilibrium preset."""
        s = Set()
        s.wing_disc_equilibrium()
        self.assertTrue(s.EnergyBarrierAR)
        self.assertEqual(s.lambdaV, 1)
        self.assertTrue(s.Remodelling)
        self.assertFalse(s.ablation)

    def test_update_derived_parameters(self):
        """Test update_derived_parameters computes dependent values."""
        s = Set()
        s.wing_disc()
        s.update_derived_parameters()
        self.assertIsNotNone(s.dt)
        self.assertIsNotNone(s.OutputFolder)
        self.assertIsNotNone(s.SubstrateZ)

    def test_check_for_non_used_parameters_energy_barrier_false(self):
        """Test check_for_non_used_parameters when EnergyBarrierA is False."""
        s = Set()
        s.EnergyBarrierA = False
        s.EnergyBarrierAR = False
        s.Bending = False
        s.Contractility = False
        s.Contractility_external = False
        s.brownian_motion = False
        s.implicit_method = False
        s.Remodelling = False
        s.check_for_non_used_parameters()
        self.assertEqual(s.lambdaB, 0)
        self.assertEqual(s.lambdaR, 0)
        self.assertEqual(s.lambdaBend, 0)
        self.assertEqual(s.cLineTension, 0)
        self.assertEqual(s.brownian_motion_scale, 0)
        self.assertEqual(s.RemodelStiffness, 2)

    def test_check_for_non_used_parameters_energy_barrier_true(self):
        """Test check_for_non_used_parameters when features are enabled."""
        s = Set()
        s.EnergyBarrierA = True
        s.EnergyBarrierAR = True
        s.Bending = True
        s.Contractility = True
        s.Contractility_external = True
        s.brownian_motion = True
        s.implicit_method = True
        s.Remodelling = True
        s.check_for_non_used_parameters()
        # When True, values are NOT set to 0
        self.assertNotEqual(s.lambdaB, 0)
        self.assertEqual(s.RemodelStiffness, 0.9)

    def test_define_if_not_defined(self):
        """Test define_if_not_defined sets values when not already set."""
        s = Set()
        s.define_if_not_defined('test_param', 42)
        self.assertEqual(s.test_param, 42)

    def test_define_if_not_defined_does_not_overwrite(self):
        """Test define_if_not_defined doesn't overwrite existing values."""
        s = Set()
        s.test_param = 100
        s.define_if_not_defined('test_param', 42)
        self.assertEqual(s.test_param, 100)

    def test_define_if_not_defined_overwrites_none(self):
        """Test define_if_not_defined overwrites None values."""
        s = Set()
        s.test_param = None
        s.define_if_not_defined('test_param', 42)
        self.assertEqual(s.test_param, 42)

    def test_copy(self):
        """Test Set.copy() creates independent copy."""
        s = Set()
        s.wing_disc()
        s2 = s.copy()
        # Modify original
        s.lambdaV = 99
        # Copy should not be modified
        self.assertNotEqual(s2.lambdaV, 99)

    def test_wound_default(self):
        """Test wound_default sets contractility parameters."""
        s = Set()
        s.Contractility = False
        s.wound_default()
        self.assertTrue(s.Contractility)

    def test_wound_default_with_multiplier(self):
        """Test wound_default with myosin_pool_multiplier."""
        s1 = Set()
        s1.wound_default(myosin_pool_multiplier=1)
        s2 = Set()
        s2.wound_default(myosin_pool_multiplier=2)
        # s2 should have a larger myosin_pool
        self.assertGreater(s2.myosin_pool, s1.myosin_pool)
