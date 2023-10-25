import numpy as np
    
def SolveRemodelingStep(Geo_0 = None,Geo_n = None,Geo = None,Dofs = None,Set = None): 
    # This function solves local problem to obtain the position of the newly
# remodeled vertices with prescribed settings (Set.***_LP), e.g.
# Set.lambda_LP.
    
    # Remark: The convergence strategy (regularization with viscosity) is
# initiated from the first iteration by setting (Set.nu_LP_Inital>Set.nu)
# and then it is reduced progressively. The solution is considered to be
# converged only when the prescribed value of global viscosity (Set.nu) is reached.
    Geop = Geo
    Geo.log = sprintf('%s\n =====>> Solving Local Problem....\n',Geo.log)
    Geo.Remodelling = True
    IncreaseEta = True
    original_nu = Set.nu
    Set.nu0 = Set.nu
    Set.nu = Set.nu_LP_Initial
    Set.MaxIter = Set.MaxIter0 * 3
    while 1:

        g,K = KgGlobal(Geo_0,Geo_n,Geo,Set)
        dy = np.zeros(((Geo.numF + Geo.numY + Geo.nCells) * 3,(Geo.numF + Geo.numY + Geo.nCells) * 3))
        dyr = norm(dy(Dofs.Remodel))
        gr = norm(g(Dofs.Remodel))
        Geo.log = sprintf('%s\n Local Problem ->Iter: %i, ||gr||= %.3e ||dyr||= %.3e  nu/nu0=%.3e  dt/dt0=%.3g \n',Geo.log,0,gr,dyr,Set.nu / Set.nu0,Set.dt / Set.dt0)
        Geo,g,K,Energy,Set,gr,dyr,dy = NewtonRaphson(Geo_0,Geo_n,Geo,Dofs,Set,K,g,- 1,- 1)
        if IncreaseEta and (gr > Set.tol or dyr > Set.tol):
            Geo = Geop
            Geo.log = sprintf('%s\n Convergence was not achieved ...\n',Geo.log)
            Geo.log = sprintf('%s\n First strategy ---> Restart iterating while higher viscosity... \n',Geo.log)
            Set.nu = Set.nu * 10
            Set.MaxIter = Set.MaxIter0 * 4
            IncreaseEta = False
        else:
            if gr > Set.tol or dyr > Set.tol or np.any(np.isnan(g(Dofs.Free))) or np.any(np.isnan(dy(Dofs.Free))):
                Geo.log = sprintf('%s\n Local Problem did not converge after %i iterations.\n',Geo.log,Set.iter)
                DidNotConverge = True
                Set.MaxIter = Set.MaxIter0
                Set.nu = original_nu
                break
            else:
                if Set.nu / Set.nu0 == 1:
                    Geo.log = sprintf('%s\n =====>> Local Problem converged in %i iterations.\n',Geo.log,Set.iter)
                    DidNotConverge = False
                    Set.MaxIter = Set.MaxIter0
                    Set.nu = original_nu
                    Geo.Remodelling = False
                    break
                else:
                    Set.nu = np.amax(Set.nu / 2,Set.nu0)

    
    return Geo,Set,DidNotConverge
    
    return Geo,Set,DidNotConverge