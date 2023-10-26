    ## Reset gradient/forces
    Geo.Cells[c].Faces[f].Tris.ContractileG = deal(0)
    
    
    Geo.Cells(c).Area = ComputeCellArea(Geo.Cells(c))
    Geo.Cells(c).Vol = ComputeCellVolume(Geo.Cells(c))
    
    return Geo
    
    return Geo