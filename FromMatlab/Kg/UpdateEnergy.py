import numpy as np
    
def UpdateEnergy(Set = None,incr = None,Energy = None): 
    # Update different energy values
    if incr == 0:
        Energies.S = np.zeros((Set.Nincr,1))
        Energies.V = np.zeros((Set.Nincr,1))
        Energies.F = np.zeros((Set.Nincr,1))
        Energies.b = np.zeros((Set.Nincr,1))
        Energies.B = np.zeros((Set.Nincr,1))
        Energies.C = np.zeros((Set.Nincr,1))
        Energies.I = np.zeros((Set.Nincr,1))
        Energies.Sub = np.zeros((Set.Nincr,1))
        Energy.Es = 0
        Energy.Ev = 0
        Energy.Ef = 0
        Energy.Eb = 0
        Energy.EB = 0
        Energy.Ec = 0
        Energy.Ei = 0
        Energy.Esub = 0
    else:
        if len(varargin) > 2:
            Energies.S[incr] = Energy.Es
            Energies.V[incr] = Energy.Ev
            Energies.B[incr] = Energy.EB
            if Set.Bending:
                Energies.b[incr] = Energy.Eb
            Energies.F[incr] = Energy.Ef
            if Set.Contractility and (Set.cPurseString > 0 or Set.cLateralCables > 0):
                Energies.C[incr] = Energy.Ec
            # Changed by Adria. Controversial. Currently undefined, have to see why
# is so.
#     if Set.Substrate
#         Energies.Sub(incr) = Energy.Esub;
#     end
    
    return Energies,Energy
    
    return Energies,Energy