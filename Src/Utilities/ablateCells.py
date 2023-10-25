    
def ablateCells(Geo = None,Set = None,t = None): 
    #PERFORMABLATION Summary of this function goes here
#   Detailed explanation goes here
    if Set.Ablation == True and Set.TInitAblation <= t:
        if len(Geo.cellsToAblate)==0 == 0:
            Geo.log = sprintf('%s ---- Performing ablation\n',Geo.log)
            for debrisCell in Geo.cellsToAblate.reshape(-1):
                Geo.Cells(debrisCell).AliveStatus = 0
                Geo.Cells(debrisCell).ExternalLambda = Set.lambdaSFactor_Debris
                Geo.Cells(debrisCell).InternalLambda = Set.lambdaSFactor_Debris
                Geo.Cells(debrisCell).SubstrateLambda = Set.lambdaSFactor_Debris
            Geo.cellsToAblate = []
    
    return Geo
    
    return Geo