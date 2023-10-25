# TODO FIXME; this function seems too complex for what it does ?
import numpy as np
    
def BuildYSubstrate(Cell = None,Cells = None,XgID = None,Set = None,XgSub = None): 
    Tets = Cell.T
    Y = Cell.Y
    nverts = len(Tets)
    X = np.zeros((len(Cells),3))
    for c in np.arange(1,len(Cells)+1).reshape(-1):
        X[c,:] = Cells(c).X
    
    for i in np.arange(1,nverts+1).reshape(-1):
        aux = ismember(Tets(i,:),XgSub)
        if np.abs(sum(aux)) > eps:
            XX = X(Tets(i,not ismember(Tets(i,:),XgID) ),:)
            if XX.shape[1-1] == 1:
                x = X(Tets(i,not aux ),:)
                Center = 1 / 3 * (np.sum(x, 1-1))
                vc = Center - X(Tets(i,not ismember(Tets(i,:),XgID) ),:)
                dis = norm(vc)
                dir = vc / dis
                offset = Set.f * dir
                Y[i,:] = X(Tets(i,not ismember(Tets(i,:),XgID) ),:) + offset
                Y[i,3] = Set.SubstrateZ
            else:
                if XX.shape[1-1] == 2:
                    X12 = XX(1,:) - XX(2,:)
                    ff = np.sqrt(Set.f ** 2 - (norm(X12) / 2) ** 2)
                    XX = np.sum(XX, 1-1) / 2
                    Center = 1 / 3 * (np.sum(X(Tets(i,not ismember(Tets(i,:),XgSub) ),:), 1-1))
                    vc = Center - XX
                    dis = norm(vc)
                    dir = vc / dis
                    offset = ff * dir
                    Y[i,:] = XX + offset
                    Y[i,3] = Set.SubstrateZ
                else:
                    if XX.shape[1-1] == 3:
                        Y[i,:] = np.multiply((1 / 3),(np.sum(X(Tets(i,not ismember(Tets(i,:),XgSub) ),:), 1-1)))
                        Y[i,3] = Set.SubstrateZ
    
    return Y
    
    return Y