"""
Test for vertex valence checking to prevent 4-fold vertices.
This test validates that the check_vertex_valence function correctly identifies
vertices that are shared by more than 3 cells.
"""
import numpy as np
import unittest
from src.pyVertexModel.geometry.geo import check_vertex_valence, Geo
from src.pyVertexModel.geometry.cell import Cell


class TestVertexValence(unittest.TestCase):
    """Test vertex valence checking functionality"""
    
    def test_check_vertex_valence_no_issues(self):
        """Test that no issues are found in a proper 3-fold vertex configuration"""
        # Create a simple geometry with proper 3-fold vertices
        geo = Geo()
        geo.nCells = 3
        geo.Cells = []
        
        # Create 3 cells that share a vertex properly (3-fold)
        # Vertex 10 is shared by cells 0, 1, 2
        for i in range(3):
            cell = Cell()
            cell.ID = i
            cell.AliveStatus = 1
            cell.T = np.array([[10, 11, 12, 13]])  # Simple tet
            cell.Y = np.array([[0.0, 0.0, 0.0]])
            geo.Cells.append(cell)
        
        # Check vertex valence - should find no issues
        problematic = check_vertex_valence(geo, log_warnings=False)
        self.assertEqual(len(problematic), 0, "Should not find any 4-fold vertices in proper 3-fold configuration")
    
    def test_check_vertex_valence_4_fold_detected(self):
        """Test that 4-fold vertices are correctly detected"""
        # Create a geometry with a 4-fold vertex (problematic)
        geo = Geo()
        geo.nCells = 4
        geo.Cells = []
        
        # Create 4 cells that all share vertex 10 (4-fold - problematic!)
        for i in range(4):
            cell = Cell()
            cell.ID = i
            cell.AliveStatus = 1
            cell.T = np.array([[10, 11 + i, 12 + i, 13 + i]])  # Each tet shares vertex 10
            cell.Y = np.array([[0.0, 0.0, 0.0]])
            geo.Cells.append(cell)
        
        # Check vertex valence - should find vertex 10 as problematic
        problematic = check_vertex_valence(geo, log_warnings=False)
        self.assertGreater(len(problematic), 0, "Should detect 4-fold vertex")
        self.assertIn(10, problematic, "Vertex 10 should be flagged as problematic")
        self.assertEqual(problematic[10]['num_cells'], 4, "Vertex 10 should be shared by 4 cells")
    
    def test_check_vertex_valence_with_dead_cells(self):
        """Test that dead cells are ignored in valence checking"""
        geo = Geo()
        geo.nCells = 4
        geo.Cells = []
        
        # Create 3 alive cells and 1 dead cell
        for i in range(4):
            cell = Cell()
            cell.ID = i
            cell.AliveStatus = 1 if i < 3 else None  # Last cell is dead
            cell.T = np.array([[10, 11 + i, 12 + i, 13 + i]])
            cell.Y = np.array([[0.0, 0.0, 0.0]])
            geo.Cells.append(cell)
        
        # Should not find issues since dead cell is ignored
        problematic = check_vertex_valence(geo, log_warnings=False)
        self.assertEqual(len(problematic), 0, "Dead cells should be ignored in valence checking")


if __name__ == '__main__':
    unittest.main()
