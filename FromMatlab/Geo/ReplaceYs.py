import numpy as np
    
def ReplaceYs(targetTets = None,Tnew = None,Ynew = None,Geo = None): 
    targetNodes = unique(targetTets)
    for n_i in np.arange(1,len(targetNodes)+1).reshape(-1):
        tNode = targetNodes(n_i)
        CellJ = Geo.Cells(tNode)
        hits = find(np.sum(ismember(CellJ.T,targetTets), 2-1) == 4)
        Geo.Cells[tNode].T[hits,:] = []
        news = find(np.sum(ismember(Tnew,tNode) == 1, 2-1))
        Geo.Cells[tNode].T[np.arange[end() + 1,end() + len[news]+1],:] = Tnew(news,:)
        if not ismember(tNode,Geo.XgID) :
            Geo.Cells[tNode].Y[hits,:] = []
            Geo.Cells[tNode].Y[np.arange[end() + 1,end() + len[news]+1],:] = Ynew(news,:)
    
    return Geo
    
    return Geo