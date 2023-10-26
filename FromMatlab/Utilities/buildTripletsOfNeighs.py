import numpy as np
    
def buildTripletsOfNeighs(neighbours = None): 
    #This method return triangles of neighs
    
    tripletsOfNeighs = []
    for i in np.arange(1,len(neighbours)+1).reshape(-1):
        neigh_cell = neighbours[i]
        for j in np.arange(1,len(neigh_cell)+1).reshape(-1):
            if neigh_cell(j) > i:
                neigh_J = neighbours[neigh_cell(j)]
                for k in np.arange(1,len(neigh_J)+1).reshape(-1):
                    if (neigh_J(k) > neigh_cell(j)):
                        common_cell = intersect(i,intersect(neigh_J,neighbours[neigh_J(k)]))
                        if (len(common_cell)==0 == 0):
                            triangleSeed = __builtint__.sorted(np.array([i,neigh_cell(j),neigh_J(k)]))
                            tripletsOfNeighs = np.array([[tripletsOfNeighs],[triangleSeed]])
    
    tripletsOfNeighs = unique(__builtint__.sorted(tripletsOfNeighs,2),'rows')
    return tripletsOfNeighs
    
    return tripletsOfNeighs