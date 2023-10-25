import numpy as np
    
def buildQuartetsOfNeighs2D(neighbours = None): 
    quartetsOfNeighs = np.array([])
    for nCell in np.arange(1,len(neighbours)+1).reshape(-1):
        neighCell = neighbours[nCell]
        interceptCells = cell(len(neighCell),1)
        for cellJ in np.arange(1,len(neighCell)+1).reshape(-1):
            commonCells = intersect(neighCell,neighbours[neighCell(cellJ)])
            if len(commonCells) > 2:
                interceptCells[cellJ] = np.array([np.transpose(commonCells),neighCell(cellJ),nCell])
        interceptCells = interceptCells(cellfun(lambda x = None: not len(x)==0 ,interceptCells))
        if not len(interceptCells)==0 :
            for indexA in np.arange(1,len(interceptCells) - 1+1).reshape(-1):
                for indexB in np.arange(indexA + 1,len(interceptCells)+1).reshape(-1):
                    intersectionCells = intersect(interceptCells[indexA],interceptCells[indexB])
                    if len(intersectionCells) >= 4:
                        quartetsOfNeighs[end() + 1,1] = nchoosek(intersectionCells,4)
    
    quartetsOfNeighs = unique(__builtint__.sorted(cell2mat(quartetsOfNeighs),2),'rows')
    return quartetsOfNeighs
    
    return quartetsOfNeighs