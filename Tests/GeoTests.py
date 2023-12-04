import numpy as np
    
def GeoTests(Geo = None): 
    #GEOTESTS Summary of this function goes here
#   Detailed explanation goes here
    
    minErrorEdge = 1e-05
    minErrorArea = minErrorEdge ** 2
    minErrorVolume = minErrorEdge ** 3
    ## Test Cells properties:
# - Volume > 0
# - Volume0 > 0
# - Area > 0
# - Area0 > 0
    for cell in Geo.Cells.reshape(-1):
        if not cell.AliveStatus==np.array([])  and not cell.AliveStatus==[] :
            assert_(cell.Vol > minErrorVolume)
            assert_(cell.Vol0 > minErrorVolume)
            assert_(cell.Area > minErrorArea)
            assert_(cell.Area0 > minErrorArea)
    
    ## Test Faces properties:
# - Area > 0
# - Area0 > 0
    for cell in Geo.Cells.reshape(-1):
        if not cell.AliveStatus==np.array([])  and not cell.AliveStatus==[] :
            for face in cell.Faces.reshape(-1):
                assert_(face.Area > minErrorArea)
                assert_(face.Area0 > minErrorArea)
    
    ## Test Tris properties:
# - Edge length > 0
# - Any LengthsToCentre > 0
# - Area > 0
    for cell in Geo.Cells.reshape(-1):
        if not cell.AliveStatus==np.array([])  and not cell.AliveStatus==[] :
            for face in cell.Faces.reshape(-1):
                for tris in face.Tris.reshape(-1):
                    assert_(tris.EdgeLength > minErrorEdge)
                    assert_(np.any(tris.LengthsToCentre > minErrorEdge))
                    assert_(tris.Area > minErrorArea)
    
    return
    