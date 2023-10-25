import numpy as np
    
def BuildInterfaceType(ij = None,XgID = None,XgTop = None,XgBottom = None): 
    valueset = np.arange(0,2+1)
    catnames = np.array(['Top','CellCell','Bottom'])
    categoricalValues = categorical(np.arange(0,2+1),valueset,catnames,'Ordinal',True)
    if np.any(ismember(ij,XgID)):
        if np.any(ismember(ij,XgTop)):
            ftype = categoricalValues(1)
        else:
            if np.any(ismember(ij,XgBottom)):
                ftype = categoricalValues(3)
            else:
                ftype = categoricalValues(2)
    else:
        ftype = categoricalValues(2)
    
    return ftype
    
    return ftype