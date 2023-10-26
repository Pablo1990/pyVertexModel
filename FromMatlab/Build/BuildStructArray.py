import numpy as np
    
def BuildStructArray(n = None,fields = None): 
    #######################################################################
# BuildStructArray:
#   Creates an array of n structs, each with fields. MATLAB imposes
#   that all structs have the same fields on creation. That is, if
#   we were to append an empty struct to CellStr, MATLAB errors out.
# Input:
#   n       : Length of the struct array
#   fields  : Fields each struct of the array will have
# Output:
#   CellStr : Array of structs
#######################################################################
    
    CellStr = struct()
    for f in np.arange(1,len(fields)+1).reshape(-1):
        setattr(CellStr,fields(f),np.array([]))
    
    for c in np.arange(2,n+1).reshape(-1):
        temp_str = struct()
        for f in np.arange(1,len(fields)+1).reshape(-1):
            setattr(temp_str,fields(f),np.array([]))
        CellStr[c] = temp_str
    
    return CellStr
    
    return CellStr