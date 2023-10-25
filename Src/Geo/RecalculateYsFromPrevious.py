import numpy as np
    
def RecalculateYsFromPrevious(Geo = None,Tnew = None,mainNodesToConnect = None,Set = None): 
    #RECALCULATEYS Summary of this function goes here
#   Detailed explanation goes here
    
    allTs = vertcat(Geo.Cells(not cellfun(isempty,np.array([Geo.Cells.AliveStatus])) ).T)
    allYs = vertcat(Geo.Cells(not cellfun(isempty,np.array([Geo.Cells.AliveStatus])) ).Y)
    nGhostNodes_allTs = np.sum(ismember(allTs,Geo.XgID), 2-1)
    Ynew = []
    possibleDebrisCells = np.array([Geo.Cells(not cellfun(isempty,np.array([Geo.Cells.AliveStatus])) ).AliveStatus]) == 0
    if np.any(possibleDebrisCells):
        debrisCells = np.array([Geo.Cells(possibleDebrisCells).ID])
    else:
        debrisCells = - 1
    
    for numTet in np.arange(1,Tnew.shape[1-1]+1).reshape(-1):
        mainNode_current = mainNodesToConnect(ismember(mainNodesToConnect,Tnew(numTet,:)))
        nGhostNodes_cTet = sum(ismember(Tnew(numTet,:),Geo.XgID))
        YnewlyComputed = ComputeY(Geo,Tnew(numTet,:),Geo.Cells(mainNode_current(1)).X,Set)
        if np.any(ismember(Tnew(numTet,:),debrisCells)):
            contributionOldYs = 1
        else:
            contributionOldYs = Set.contributionOldYs
        if np.all(not ismember(Tnew(numTet,:),np.array([Geo.XgBottom,Geo.XgTop])) ):
            Ynew[end() + 1,:] = YnewlyComputed
        else:
            tetsToUse = np.sum(ismember(allTs,Tnew(numTet,:)), 2-1) > 2
            if np.any(ismember(Tnew(numTet,:),Geo.XgTop)):
                tetsToUse = np.logical_and(tetsToUse,np.any(ismember(allTs,Geo.XgTop),2))
            else:
                if np.any(ismember(Tnew(numTet,:),Geo.XgBottom)):
                    tetsToUse = np.logical_and(tetsToUse,np.any(ismember(allTs,Geo.XgBottom),2))
            tetsToUse = np.logical_and(tetsToUse,nGhostNodes_allTs) == nGhostNodes_cTet
            if np.any(tetsToUse):
                Ynew[end() + 1,:] = contributionOldYs * mean(vertcat(allYs(tetsToUse,:)),1) + (1 - contributionOldYs) * YnewlyComputed
            else:
                tetsToUse = np.sum(ismember(allTs,Tnew(numTet,:)), 2-1) > 1
                if np.any(ismember(Tnew(numTet,:),Geo.XgTop)):
                    tetsToUse = np.logical_and(tetsToUse,np.any(ismember(allTs,Geo.XgTop),2))
                else:
                    if np.any(ismember(Tnew(numTet,:),Geo.XgBottom)):
                        tetsToUse = np.logical_and(tetsToUse,np.any(ismember(allTs,Geo.XgBottom),2))
                tetsToUse = np.logical_and(tetsToUse,nGhostNodes_allTs) == nGhostNodes_cTet
                if np.any(tetsToUse):
                    Ynew[end() + 1,:] = contributionOldYs * mean(vertcat(allYs(tetsToUse,:)),1) + (1 - contributionOldYs) * YnewlyComputed
                else:
                    Ynew[end() + 1,:] = YnewlyComputed
    
    return Ynew
    
    return Ynew