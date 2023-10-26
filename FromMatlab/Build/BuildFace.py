import numpy as np
    
def BuildFace(ci = None,cj = None,face_ids = None,nCells = None,Cell = None,XgID = None,Set = None,XgTop = None,XgBottom = None,oldFace = None): 
    #######################################################################
# BuildFace:
#   Completes a single Face struct with already but empty fields.
#   Its fields are:
#		- ij            : Cells that have Face in contact
#       - globalIds     : globalId (UNVECTORIZED) for the face centre
#		- InterfaceType : Integer. If the face is facing the substrate
#		another cell or the lumen
#		- Centre        : Face centre
#       - Edges         : Local indices of the vertices forming the
#		face. That is Geo.Cells(c).Y(Edges(e,:),:) will give vertices
#       defining the edge. Used also for triangle computation
# Input:
#   ci   : index for cell i
#   cj   : index for cell j
#   Cell : Cell object
#   XgID : Nodal Ids for ghost nodes
#   Set  : User Defined settings
# Output:
#   Face : Face struct with filled data
#######################################################################
    
    ij = np.array([ci,cj])
    Face = struct()
    Face.ij = ij
    Face.globalIds = - 1
    Face.InterfaceType = BuildInterfaceType(ij,XgID,XgTop,XgBottom)
    newFaceCentre = BuildFaceCentre(ij,nCells,Cell.X,Cell.Y(face_ids,:),Set.f,Set.InputGeo=='Bubbles')
    if ('oldFace' is not None) and not len(oldFace)==0 :
        Face.Centre = oldFace.Centre
        #Face.Tris = oldFace.Tris;
    else:
        Face.Centre = newFaceCentre
    
    ## DON'T KNOW WHY BUT I HAVE TO CREATE THE TRIS ALL THE TIME (CAN'T USE THE OLD FACE TRIS)
    Face.Tris = BuildEdges(Cell.T,face_ids,Face.Centre,Face.InterfaceType,Cell.X,Cell.Y,np.arange(1,nCells+1))
    
    Face.Area = ComputeFaceArea(vertcat(Face.Tris.Edge),Cell.Y,Face.Centre)
    Face.Area0 = Face.Area
    return Face
    
    return Face