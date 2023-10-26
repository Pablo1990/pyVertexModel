import numpy as np
    
def KgBulkElem(x = None,x0 = None,mu = None,lambda_ = None,Neo = None): 
    #KGBULKELEM Computes elemental residual and Jacobian for bulk viscoelastic domain
#   Can use:
#   Neo=0: St. Venant-Kirchoff elastic poential:
#   W(E)=lambda tr(E)^2 + 2*mu tr(E^2)
    
    #   Neo=1: Neo-Hookean elastic poential:
#   W(E)=lambda ln(J)^2 + mu (tr(E)-ln(J))
    
    #   Neo=2: Modififed Neo-Hookean for handling negative volumes (may happen in iterative process)
#   W(E)=lambda ln(J^2)^2 + mu (tr(E)-ln(J^2))
#   If no lambda is given, assumed to be zero (no volumetric stiffness)
    
    #   INPUT:
#   x(i,:)=coordinates of node i (assumed 4 nodes) on deformed element
#   x0(i,:)=coordinates of node i (assumed 4 nodes) on undeformed element
#   mu,lambda = material shear stiffness
    
    #   OUTPUT:
#   g=elemental residual
#   K=elemental Jacobian
#   S=2nf Piola-Kirchhof stress tensor on last GP
    
    #   Designed by Jose J. Mu�oz
    
    NeoH = 2
    if ('Neo' is not None):
        NeoH = Neo
    
    if not ('lambda' is not None) :
        lambda_ = 0
    
    if x.shape[1-1] != 4:
        raise Exception('Bulk stiffness can only be used with 4 nodal tetrahedra.')
    
    nnod,dim = x.shape
    ng = 1
    
    if ng == 1:
        xg = np.array([1,1,1]) / 4
        wg = 1
    else:
        if ng == 4:
            a = 0.5854102
            b = (1 - a) / 3
            xg = np.array([b,b,b,a,b,b,b,a,b,b,b,a])
            wg = np.array([1,1,1,1]) / 4
    
    wg = wg / 6
    
    # Derivatives are constant
    DN = np.array([1,0,0,- 1,0,1,0,- 1,0,0,1,- 1])
    
    g = np.zeros((nnod * dim,1))
    K = np.zeros((nnod * dim,nnod * dim))
    Energy = 0
    for ig in np.arange(1,ng+1).reshape(-1):
        # Unnecessary: dxdxi=x'*DN'; # 3x3 Jacobain dx_i/dxi_j
        dXdxi = np.transpose(x0) * np.transpose(DN)
        gradXN = np.linalg.solve((np.transpose(dXdxi)),DN)
        F = np.transpose(x) * np.transpose(gradXN)
        gradxN = np.linalg.solve((np.transpose(F)),gradXN)
        J = det(F)
        E = 0.5 * (np.transpose(F) * F - np.eye(3))
        trE = sum(diag(E))
        Je = det(dXdxi)
        lJ = np.log(J)
        lJ2 = np.log(J ** 2)
        if J < 0:
            #         warning('Inverted tetrahedra');
#         ME = MException('KgBulkElem:invertedTetrahedralElement', ...
#             'Inverted Tetrahedral Element');
#         throw(ME)
            pass
        if NeoH != 2 and Je < 0:
            raise Exception('Tetrahedral Element orientation need to be swapped')
        if NeoH == 1:
            Energy = Energy + 0.5 * lambda_ * lJ ** 2 + mu * (trE - lJ) * Je * wg(ig)
        else:
            if NeoH == 2:
                Energy = Energy + 0.5 * lambda_ * lJ2 ** 2 + mu * (trE - lJ2) * Je * wg(ig)
            else:
                Energy = Energy + 0.5 * lambda_ * trE ** 2 + mu * sum(diag(E * E)) * Je * wg(ig)
        for a in np.arange(1,nnod+1).reshape(-1):
            gradXNa = gradXN(:,a)
            gradxNa = gradxN(:,a)
            idof = np.arange((a - 1) * dim + 1,a * dim+1)
            if NeoH == 1:
                K1 = (lambda_ * lJ - mu) * gradxNa + mu * F * gradXNa
            else:
                if NeoH == 2:
                    K1 = 2 * (lambda_ * lJ2 - mu) * gradxNa + mu * F * gradXNa
                else:
                    K1 = (lambda_ * trE * F + 2 * mu * F * E) * gradXNa
            g[idof] = g(idof) + wg(ig) * K1 * Je
            for b in np.arange(1,nnod+1).reshape(-1):
                gradXNb = gradXN(:,b)
                gradxNb = gradxN(:,b)
                jdof = np.arange((b - 1) * dim + 1,b * dim+1)
                NxaNxb = gradxNa * np.transpose(gradxNb)
                if NeoH == 1:
                    K1 = mu * (np.transpose(gradXNa) * gradXNb) * np.eye(3)
                    K2 = lambda_ * NxaNxb
                    K3 = - (lambda_ * lJ - mu) * np.transpose(NxaNxb)
                    Kab = K1 + K2 + K3
                else:
                    if NeoH == 2:
                        K1 = mu * (np.transpose(gradXNa) * gradXNb) * np.eye(3)
                        K2 = 4 * lambda_ * NxaNxb
                        K3 = - 2 * (lambda_ * lJ2 - mu) * np.transpose(NxaNxb)
                        Kab = K1 + K2 + K3
                    else:
                        K1 = (trE * lambda_ * (np.transpose(gradXNa) * gradXNb) + 2 * mu * (np.transpose(gradXNb) * E * gradXNa)) * np.eye(3)
                        K2 = lambda_ * F * gradXNa * np.transpose((F * gradXNb)) + mu * (F * gradXNb * np.transpose((F * gradXNa)))
                        K3 = mu * F * (np.transpose(F)) * (np.transpose(gradXNa) * gradXNb)
                        Kab = K1 + K2 + K3
                K[idof,jdof] = K(idof,jdof) + wg(ig) * Je * Kab
    
    return g,K,Energy
    
    return g,K,Energy