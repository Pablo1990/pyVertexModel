import numpy as np
    
def AddTetrahedra(Geo = None,oldGeo = None,newTets = None,Ynew = None,Set = None): 
    #ADDTETRAHEDRA Summary of this function goes here
#   Detailed explanation goes here
    
    if not ('Ynew' is not None) :
        Ynew = []
    
    for newTet in np.transpose(newTets).reshape(-1):
        if np.any(not ismember(newTet,Geo.XgID) ):
            for numNode in np.transpose(newTet).reshape(-1):
                if not np.any(ismember(newTet,Geo.XgID))  and ismember(np.transpose(__builtint__.sorted(newTet)),__builtint__.sorted(Geo.Cells(numNode).T,2),'rows'):
                    Geo.Cells[numNode].Y[ismember[__builtint__.sorted[Geo.Cells[numNode].T,2],np.transpose[__builtint__.sorted[newTet]],'rows'],:] = []
                    Geo.Cells[numNode].T[ismember[__builtint__.sorted[Geo.Cells[numNode].T,2],np.transpose[__builtint__.sorted[newTet]],'rows'],:] = []
                else:
                    if len(Geo.Cells(numNode).T)==0 or not ismember(np.transpose(__builtint__.sorted(newTet)),__builtint__.sorted(Geo.Cells(numNode).T,2),'rows') :
                        #DT = delaunayTriangulation(vertcat(Geo.Cells(newTet).X));
#if ~any(ismember(newTet, Geo.XgID))
                        Geo.Cells[numNode].T[end() + 1,:] = newTet
                        #else
#Geo.Cells(numNode).T(end+1, :) = newTet(DT.ConnectivityList);
#end
                        if not len(Geo.Cells(numNode).AliveStatus)==0  and ('Set' is not None):
                            if not len(Ynew)==0 :
                                Geo.Cells[numNode].Y[end() + 1,:] = Ynew(ismember(newTets,np.transpose(newTet),'rows'),:)
                            else:
                                Geo.Cells[numNode].Y[end() + 1,:] = RecalculateYsFromPrevious(oldGeo,np.transpose(newTet),numNode,Set)
                            Geo.numY = Geo.numY + 1
    
    return Geo
    
    return Geo