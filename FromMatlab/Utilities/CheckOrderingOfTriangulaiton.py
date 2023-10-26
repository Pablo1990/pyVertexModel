import numpy as np
import warnings
    
def CheckOrderingOfTriangulaiton(Cell = None,Y = None,Set = None): 
    ## This function makes sure that the triangulation of cell-Faces is ordered correctly.
    
    Sides = np.array([[1,2],[2,3],[3,1]])
    Recompute = False
    for c in np.array([Geo.Cells(not cellfun(isempty,np.array([Geo.Cells.AliveStatus])) ).ID]).reshape(-1):
        Cell = Geo.Cells(c)
        IsConsistent = True
        for f in np.arange(1,len(Cell.Faces)+1).reshape(-1):
            Face = Cell.Faces(f)
            Tris = Face.Tris
            Tris[Tris[:,3] > 0,3] = Tris(Tris(:,3) > 0,3) + Geo.numY
            Tris[Tris[:,3] < 0,3] = np.abs(Tris(Tris(:,3) < 0,3))
            Id = np.arange(1,Tris.shape[1-1]+1)
            for t in np.arange(1,len(Tris)+1).reshape(-1):
                Side1 = Tris(t,Sides(1,:))
                TID = Id(np.sum(ismember(Tris,Side1), 2-1) == 2)
                TID[TID == t] = []
                if (Side1(1) == Tris(TID,1) and Side1(2) == Tris(TID,2)) or (Side1(1) == Tris(TID,2) and Side1(2) == Tris(TID,3)) or (Side1(1) == Tris(TID,3) and Side1(2) == Tris(TID,1)):
                    #            fprintf('Triangles #i and #i of #i cell are incompatible \n',t,TID,c);
                    IsConsistent = False
                    break
                Side2 = Tris(t,Sides(2,:))
                TID = Id(np.sum(ismember(Tris,Side2), 2-1) == 2)
                TID[TID == t] = []
                if (Side2(1) == Tris(TID,1) and Side2(2) == Tris(TID,2)) or (Side2(1) == Tris(TID,2) and Side2(2) == Tris(TID,3)) or (Side2(1) == Tris(TID,3) and Side2(2) == Tris(TID,1)):
                    #            fprintf('Triangles #i and #i of #i cell are incompatible \n',t,TID,c);
                    IsConsistent = False
                    break
                Side3 = Tris(t,Sides(3,:))
                TID = Id(np.sum(ismember(Tris,Side3), 2-1) == 2)
                TID[TID == t] = []
                if (Side3(1) == Tris(TID,1) and Side3(2) == Tris(TID,2)) or (Side3(1) == Tris(TID,2) and Side3(2) == Tris(TID,3)) or (Side3(1) == Tris(TID,3) and Side3(2) == Tris(TID,1)):
                    #            fprintf('Triangles #i and #i of #i cell are incompatible \n',t,TID,c);
                    IsConsistent = False
                    break
                if IsConsistent:
                    continue
                warnings.warn('Inconsistent triangulation was detected.... \n')
                Recompute = True
                # Choose a particular consistent order between the two options
                Included = np.zeros((Tris.shape[1-1],1))
                TrisAux = np.zeros((Tris.shape,Tris.shape))
                TrisAux[1,:] = Tris(1,:)
                Included[1] = 1
                NotAllIncluded = True
                while NotAllIncluded:

                    for t in np.arange(2,Tris.shape[1-1]+1).reshape(-1):
                        if Included(t) == 1:
                            continue
                        T = Tris(t,:)
                        for tt in np.arange(1,Tris.shape[1-1]+1).reshape(-1):
                            if tt == t or Included(tt) == 0:
                                continue
                            TT = np.array([TrisAux(tt,np.array([1,2])),Tris(tt,3)])
                            if (T(1) == TT(1) and T(2) == TT(2)) or (T(1) == TT(2) and T(2) == TT(3)) or (T(1) == TT(3) and T(2) == TT(1)) or (T(2) == TT(1) and T(3) == TT(2)) or (T(2) == TT(2) and T(3) == TT(3)) or (T(2) == TT(3) and T(3) == TT(1)) or (T(3) == TT(1) and T(1) == TT(2)) or (T(3) == TT(2) and T(1) == TT(3)) or (T(3) == TT(3) and T(1) == TT(1)):
                                TrisAux[t,:] = np.array([Cell.Tris[c](t,2),Cell.Tris[c](t,1),Cell.Tris[c](t,3)])
                                Included[t] = 1
                                break
                            else:
                                if (T(2) == TT(1) and T(1) == TT(2)) or (T(2) == TT(2) and T(1) == TT(3)) or (T(2) == TT(3) and T(1) == TT(1)) or (T(3) == TT(1) and T(2) == TT(2)) or (T(3) == TT(2) and T(2) == TT(3)) or (T(3) == TT(3) and T(2) == TT(1)) or (T(1) == TT(1) and T(3) == TT(2)) or (T(1) == TT(2) and T(3) == TT(3)) or (T(1) == TT(3) and T(3) == TT(1)):
                                    TrisAux[t,:] = Cell.Tris[c](t,:)
                                    Included[t] = 1
                                    break
                    if sum(Included) == len(Included):
                        TrisAux[1,:] = Cell.Tris[c](1,:)
                        NotAllIncluded = False

    
    # loop over Cell (Checking the order of vertices of =each triangle )
    for c in np.arange(1,Cell.n+1).reshape(-1):
        IsConsistent = True
        Tris = Cell.Tris[c]
        Tris[Tris[:,3] > 0,3] = Tris(Tris(:,3) > 0,3) + Y.n
        Tris[Tris[:,3] < 0,3] = np.abs(Tris(Tris(:,3) < 0,3))
        Id = np.arange(1,Tris.shape[1-1]+1)
        for t in np.arange(1,Tris.shape[1-1]+1).reshape(-1):
            Side1 = Tris(t,Sides(1,:))
            TID = Id(np.sum(ismember(Tris,Side1), 2-1) == 2)
            TID[TID == t] = []
            if (Side1(1) == Tris(TID,1) and Side1(2) == Tris(TID,2)) or (Side1(1) == Tris(TID,2) and Side1(2) == Tris(TID,3)) or (Side1(1) == Tris(TID,3) and Side1(2) == Tris(TID,1)):
                #            fprintf('Triangles #i and #i of #i cell are incompatible \n',t,TID,c);
                IsConsistent = False
                break
            Side2 = Tris(t,Sides(2,:))
            TID = Id(np.sum(ismember(Tris,Side2), 2-1) == 2)
            TID[TID == t] = []
            if (Side2(1) == Tris(TID,1) and Side2(2) == Tris(TID,2)) or (Side2(1) == Tris(TID,2) and Side2(2) == Tris(TID,3)) or (Side2(1) == Tris(TID,3) and Side2(2) == Tris(TID,1)):
                #            fprintf('Triangles #i and #i of #i cell are incompatible \n',t,TID,c);
                IsConsistent = False
                break
            Side3 = Tris(t,Sides(3,:))
            TID = Id(np.sum(ismember(Tris,Side3), 2-1) == 2)
            TID[TID == t] = []
            if (Side3(1) == Tris(TID,1) and Side3(2) == Tris(TID,2)) or (Side3(1) == Tris(TID,2) and Side3(2) == Tris(TID,3)) or (Side3(1) == Tris(TID,3) and Side3(2) == Tris(TID,1)):
                #            fprintf('Triangles #i and #i of #i cell are incompatible \n',t,TID,c);
                IsConsistent = False
                break
        if IsConsistent:
            continue
        warnings.warn('Inconsistent triangulation was detected.... \n')
        Recompute = True
        # Choose a particular consistent order between the two options
        Included = np.zeros((Tris.shape[1-1],1))
        TrisAux = np.zeros((Tris.shape,Tris.shape))
        TrisAux[1,:] = Tris(1,:)
        Included[1] = 1
        NotAllIncluded = True
        while NotAllIncluded:

            for t in np.arange(2,Tris.shape[1-1]+1).reshape(-1):
                if Included(t) == 1:
                    continue
                T = Tris(t,:)
                for tt in np.arange(1,Tris.shape[1-1]+1).reshape(-1):
                    if tt == t or Included(tt) == 0:
                        continue
                    TT = np.array([TrisAux(tt,np.array([1,2])),Tris(tt,3)])
                    if (T(1) == TT(1) and T(2) == TT(2)) or (T(1) == TT(2) and T(2) == TT(3)) or (T(1) == TT(3) and T(2) == TT(1)) or (T(2) == TT(1) and T(3) == TT(2)) or (T(2) == TT(2) and T(3) == TT(3)) or (T(2) == TT(3) and T(3) == TT(1)) or (T(3) == TT(1) and T(1) == TT(2)) or (T(3) == TT(2) and T(1) == TT(3)) or (T(3) == TT(3) and T(1) == TT(1)):
                        TrisAux[t,:] = np.array([Cell.Tris[c](t,2),Cell.Tris[c](t,1),Cell.Tris[c](t,3)])
                        Included[t] = 1
                        break
                    else:
                        if (T(2) == TT(1) and T(1) == TT(2)) or (T(2) == TT(2) and T(1) == TT(3)) or (T(2) == TT(3) and T(1) == TT(1)) or (T(3) == TT(1) and T(2) == TT(2)) or (T(3) == TT(2) and T(2) == TT(3)) or (T(3) == TT(3) and T(2) == TT(1)) or (T(1) == TT(1) and T(3) == TT(2)) or (T(1) == TT(2) and T(3) == TT(3)) or (T(1) == TT(3) and T(3) == TT(1)):
                            TrisAux[t,:] = Cell.Tris[c](t,:)
                            Included[t] = 1
                            break
            if sum(Included) == len(Included):
                TrisAux[1,:] = Cell.Tris[c](1,:)
                NotAllIncluded = False

        # compute volume
        auxV = 0
        for t in np.arange(1,TrisAux.shape[1-1]+1).reshape(-1):
            if TrisAux(t,3) < 1:
                YTri = np.array([[Y.DataRow(TrisAux(t,np.array([1,2])),:)],[Y.DataRow(np.abs(TrisAux(t,3)),:)]])
            else:
                YTri = np.array([[Y.DataRow(TrisAux(t,np.array([1,2])),:)],[Cell.FaceCentres.DataRow(TrisAux(t,3),:)]])
            T = det(YTri) / 6
            auxV = auxV + T
        # if the volume is negative switch two the other option
        if auxV < 0:
            TrisAux = np.array([TrisAux(:,2),TrisAux(:,1),TrisAux(:,3)])
        # Correct Cell and faces Data
        Cell.Tris[c] = TrisAux
        aux1 = 1
        for s in np.arange(1,Cell.Faces[c].nFaces+1).reshape(-1):
            if Cell.Faces[c].Vertices[s](1) == TrisAux(aux1,2) and Cell.Faces[c].Vertices[s](2) == TrisAux(aux1,1):
                Cell.Faces[c].Vertices[s] = flip(Cell.Faces[c].Vertices[s])
                if len(Cell.Faces[c].Vertices[s]) == 3:
                    Cell.Faces[c].Vertices[s] = np.abs(TrisAux(aux1,:))
                    Cell.Faces[c].Tris[s] = TrisAux(aux1,:)
                else:
                    Cell.Faces[c].Tris[s] = TrisAux(np.arange(aux1,aux1 + len(Cell.Faces[c].Vertices[s]) - 1+1),:)
                Cell.AllFaces.Vertices[Cell.Faces[c].FaceCentresID[s]] = Cell.Faces[c].Vertices[s]
            if len(Cell.Faces[c].Vertices[s]) == 3:
                aux1 = aux1 + 1
            else:
                aux1 = aux1 + len(Cell.Faces[c].Vertices[s])
    
    # Recompte Volume and Surface Area (This can be improved)
    if Recompute:
        Cell = BuildEdges(Cell,Y)
        Cell = ComputeCellVolume(Cell,Y)
        Cell.AllFaces = Cell.AllFaces.ComputeAreaTri(Y.DataRow,Cell.FaceCentres.DataRow)
        Cell.AllFaces = Cell.AllFaces.ComputeEnergy(Set)
    
    return Cell,Y
    
    return Cell,Y