import numpy as np
    
def KgBending(Cell = None,Y = None,Set = None): 
    # The residual g and Jacobian K of the Bending Energy
#  Potential: -> Set.BendingAreaDependent=1 : Wb=(1/2) lambdaBend* sum_edges( 1-cos(theta/2)^2*(At1+At2)
#             -> Set.BendingAreaDependent=0 : Wb=(1/2) lambdaBend* sum_edges( 1-cos(theta/2)^2
#                       where  theta: the angle between the pair of triangles
#                              At1 and At2 : the area of the triangles
    
    ## Initialize
    if nargout > 1:
        if Set.Sparse == 2:
            g,Energy,ncell,K,si,sj,sk,sv = initializeKg(Cell,Set)
        else:
            g,Energy,ncell,K = initializeKg(Cell,Set)
    else:
        g,Energy,ncell = initializeKg(Cell,Set)
    
    ## Loop over cells
    for i in np.arange(1,ncell+1).reshape(-1):
        if Cell.DebrisCells(i):
            continue
        if not Cell.AssembleAll :
            if not ismember(Cell.Int(i),Cell.AssembleNodes) :
                continue
        L = Set.lambdaBend
        Edges = Cell.Edges[i]
        for e in np.arange(1,Edges.shape[1-1]+1).reshape(-1):
            if not Cell.AssembleAll  and not np.any(ismember(Edges(e,:),Cell.RemodelledVertices)) :
                continue
            if Edges(e,1) <= Y.n:
                Y1 = Y.DataRow(Edges(e,1),:)
            else:
                Y1 = Cell.FaceCentres.DataRow(Edges(e,1) - Y.n,:)
            if Edges(e,2) <= Y.n:
                Y2 = Y.DataRow(Edges(e,2),:)
            else:
                Y2 = Cell.FaceCentres.DataRow(Edges(e,2) - Y.n,:)
            if Edges(e,3) <= Y.n:
                Y3 = Y.DataRow(Edges(e,3),:)
            else:
                Y3 = Cell.FaceCentres.DataRow(Edges(e,3) - Y.n,:)
            if Edges(e,4) <= Y.n:
                Y4 = Y.DataRow(Edges(e,4),:)
            else:
                Y4 = Cell.FaceCentres.DataRow(Edges(e,4) - Y.n,:)
            n1 = cross(Y2 - Y1,Y3 - Y1)
            n2 = cross(Y4 - Y1,Y2 - Y1)
            A1 = (1 / 2) * norm(n1)
            A2 = (1 / 2) * norm(n2)
            B = np.dot(n1,n2)
            n1 = n1 / norm(n1)
            n2 = n2 / norm(n2)
            nn = np.dot(n1,n2)
            if nn == - 1:
                nn = 1
            fact0 = (L / 12) * (1 - 2 / np.sqrt(2 * nn + 2))
            fact1 = (L / 6) * (1 - np.sqrt(2 * nn + 2) / 2) ** 2
            fact2 = (L / 6) * (1 / ((2 * nn + 2) ** (3 / 2)))
            dA1dy,dA2dy,dA1dydy,dA2dydy = dAdY(Y1,Y2,Y3,Y4)
            dBdy,dBdydy = dBdY(Y1,Y2,Y3,Y4)
            dAdy = dA1dy + dA2dy
            if np.abs(B) < eps:
                S = np.zeros((dBdy.shape,dBdy.shape))
                dnndy = np.zeros((S.shape,S.shape))
                Knn = np.zeros((len(S),len(S)))
            else:
                S = ((1 / B) * dBdy - (1 / A1) * dA1dy - (1 / A2) * dA2dy)
                dnndy = nn * S
                Knn = nn * (S * np.transpose(S) + (1 / B) * dBdydy - (1 / B ** 2) * dBdy * (np.transpose(dBdy)) - (1 / A1) * dA1dydy + (1 / A1 ** 2) * dA1dy * (np.transpose(dA1dy)) - (1 / A2) * dA2dydy + (1 / A2 ** 2) * dA2dy * (np.transpose(dA2dy)))
            if Set.BendingAreaDependent:
                ge = fact0 * (A1 + A2) * dnndy + fact1 * dAdy
                Ke = fact2 * (A1 + A2) * dnndy * (np.transpose(dnndy)) + fact0 * ((A1 + A2) * Knn + dnndy * np.transpose(dAdy) + dAdy * np.transpose(dnndy)) + fact1 * (dA1dydy + dA2dydy)
            else:
                # Area independent
                ge = fact0 * dnndy
                Ke = fact2 * dnndy * (np.transpose(dnndy)) + fact0 * Knn
            # Assembleg
            dim = 3
            for I in np.arange(np.arange(1,len(Edges(e,,))+1)):
                idofg = np.arange((Edges(e,I) - 1) * dim + 1,Edges(e,I) * dim+1)
                idofl = np.arange((I - 1) * dim + 1,I * dim+1)
                g[idofg] = g(idofg) + ge(idofl)
            # AssembleK
            if nargout > 1:
                if Set.Sparse == 2:
                    si,sj,sv,sk = AssembleKSparse(Ke,Edges(e,:),si,sj,sv,sk)
                else:
                    K = AssembleK(K,Ke,Edges(e,:))
                Energy = Energy + (L / 6) * (1 - np.sqrt(2 * nn + 2) / 2) ** 2
    
    if Set.Sparse == 2 and nargout > 1:
        K = sparse(si(np.arange(1,sk+1)),sj(np.arange(1,sk+1)),sv(np.arange(1,sk+1)),K.shape[1-1],K.shape[2-1])
    
    ## Loop over Cells
#     # Analytical residual g and Jacobian K
# for i=1:ncell
#     if ~Cell.AssembleAll
#         if ~ismember(Cell.Int(i),Cell.AssembleNodes)
#            continue
#         end
#     end
#     lambdaS=Set.lambdaS;
# #     fact=( lambdaS / (4*Cell.SArea(i)) ) *(  (Cell.SArea(i)-Cell.SArea0(i)) / Cell.SArea0(i)^2   );
# if Set.A0eq0
#     fact=lambdaS *  (Cell.SArea(i)) / Cell.SArea0(i)^2   ;
# else
#     fact=lambdaS *  (Cell.SArea(i)-Cell.SArea0(i)) / Cell.SArea0(i)^2   ;
# end
#     ge=zeros(dimg,1); # Local cell residual
# #             K2=zeros(dimg); # Also used in sparse
    
    #     # Loop over Cell-face-triangles
#     Tris=Cell.Tris{i};
#     for t=1:size(Tris,1)
#         nY=Tris(t,:);
#         Y1=Y.DataRow(nY(1),:);
#         Y2=Y.DataRow(nY(2),:);
#         Y4=Cell.FaceCentres.DataRow(nY(3),:);
#         nY(3)=nY(3)+Set.NumMainV;
#         [gs,Ks,Kss]=gKSArea(Y1,Y2,Y4);
#         Ks=fact*(Ks+Kss);
#         ge=AssemblegSArea(ge,gs,nY);
#         if Set.Sparse
#             [si,sj,sv,sk]= AssembleKSparse(Ks,nY,si,sj,sv,sk);
    
    #         else
#             K= AssembleK(K,Ks,nY);
#         end
#     end
    
    #     g=g+ge*fact; # Volume contribution of each triangle is (y1-y2)'*J*(y2-y3)/2
#     if Set.Sparse
#         K=K+sparse((ge)*(ge')*lambdaS/(Cell.SArea0(i)^2));
#     else
#        K=K+(ge)*(ge')*lambdaS/(Cell.SArea0(i)^2);  #-(gee)*(gee')*(fact);
#     end
    
    #     if Set.A0eq0
#         EnergyBend=EnergyBend+ lambdaS/2 *((Cell.SArea(i)) / Cell.SArea0(i))^2;
#     else
#         EnergyBend=EnergyBend+ lambdaS/2 *((Cell.SArea(i)-Cell.SArea0(i)) / Cell.SArea0(i))^2;
#     end
    
    # end
    
    # if Set.Sparse
#     K=sparse(si(1:sk),sj(1:sk),sv(1:sk),dimg,dimg)+K;
# end
    return g,K,Cell,Energy
    
    ##
    
def dBdY(y1 = None,y2 = None,y3 = None,y4 = None): 
    y12 = np.transpose(y2) - np.transpose(y1)
    y23 = np.transpose(y3) - np.transpose(y2)
    y24 = np.transpose(y4) - np.transpose(y2)
    y13 = np.transpose(y3) - np.transpose(y1)
    y14 = np.transpose(y4) - np.transpose(y1)
    dBdy1 = Cross(y24) * Cross(y12) * y13 - Cross(y23) * Cross(y14) * y12
    dBdy2 = - Cross(y14) * Cross(y12) * y13 + Cross(y13) * Cross(y14) * y12
    dBdy3 = - Cross(y12) * Cross(y14) * y12
    dBdy4 = Cross(y12) * Cross(y12) * y13
    dBdy = np.array([[dBdy1],[dBdy2],[dBdy3],[dBdy4]])
    dBdy1dy1 = Cross(y24) * Cross(y23) + Cross(y23) * Cross(y24)
    dBdy2dy2 = Cross(y14) * Cross(y13) + Cross(y13) * Cross(y14)
    dBdy3dy3 = np.zeros((3,3))
    dBdy4dy4 = np.zeros((3,3))
    dBdy1dy2 = - Cross(y24) * Cross(y13) + Cross(Cross(y12) * y13) - Cross(y23) * Cross(y14) - Cross(Cross(y14) * y12)
    dBdy1dy3 = Cross(y24) * Cross(y12) + Cross(Cross(y14) * y12)
    dBdy1dy4 = Cross(y23) * Cross(y12) - Cross(Cross(y12) * y13)
    dBdy2dy3 = - Cross(y14) * Cross(y12) - Cross(Cross(y14) * y12)
    dBdy2dy4 = - Cross(y13) * Cross(y12) + Cross(Cross(y12) * y13)
    dBdy3dy4 = Cross(y12) * Cross(y12)
    dBdydy = np.array([[dBdy1dy1,dBdy1dy2,dBdy1dy3,dBdy1dy4],[np.transpose(dBdy1dy2),dBdy2dy2,dBdy2dy3,dBdy2dy4],[np.transpose(dBdy1dy3),np.transpose(dBdy2dy3),dBdy3dy3,dBdy3dy4],[np.transpose(dBdy1dy4),np.transpose(dBdy2dy4),np.transpose(dBdy3dy4),dBdy4dy4]])
    return dBdy,dBdydy
    
    
def dAdY(y1 = None,y2 = None,y3 = None,y4 = None): 
    q1 = Cross(y2) * np.transpose(y1) - Cross(y2) * np.transpose(y3) + Cross(y1) * np.transpose(y3)
    q2 = Cross(y2) * np.transpose(y1) - Cross(y2) * np.transpose(y4) + Cross(y1) * np.transpose(y4)
    Q11 = Cross(y2) - Cross(y3)
    Q21 = Cross(y3) - Cross(y1)
    Q12 = Cross(y2) - Cross(y4)
    Q22 = Cross(y4) - Cross(y1)
    Q3 = Cross(y1) - Cross(y2)
    fact1 = 1 / (2 * norm(q1))
    fact2 = 1 / (2 * norm(q2))
    ga1 = np.array([[fact1 * np.transpose(Q11) * q1],[fact1 * np.transpose(Q21) * q1],[fact1 * np.transpose(Q3) * q1],[np.zeros((3,1))]])
    ga2 = np.array([[fact2 * np.transpose(Q12) * q2],[fact2 * np.transpose(Q22) * q2],[np.zeros((3,1))],[fact2 * np.transpose(Q3) * q2]])
    Ka1 = np.multiply(- (2 / norm(q1)),(ga1)) * (np.transpose(ga1))
    Ka2 = np.multiply(- (2 / norm(q2)),(ga2)) * (np.transpose(ga2))
    Kaa1 = np.multiply(fact1,np.array([[np.transpose(Q11) * Q11,KK(1,2,3,y1,y2,y3),KK(1,3,2,y1,y2,y3),np.zeros((3,3))],[KK(2,1,3,y1,y2,y3),np.transpose(Q21) * Q21,KK(2,3,1,y1,y2,y3),np.zeros((3,3))],[KK(3,1,2,y1,y2,y3),KK(3,2,1,y1,y2,y3),np.transpose(Q3) * Q3,np.zeros((3,3))],[np.zeros((3,3)),np.zeros((3,3)),np.zeros((3,3)),np.zeros((3,3))]]))
    Kaa2 = np.multiply(fact2,np.array([[np.transpose(Q12) * Q12,KK(1,2,3,y1,y2,y4),np.zeros((3,3)),KK(1,3,2,y1,y2,y4)],[KK(2,1,3,y1,y2,y4),np.transpose(Q22) * Q22,np.zeros((3,3)),KK(2,3,1,y1,y2,y4)],[np.zeros((3,3)),np.zeros((3,3)),np.zeros((3,3)),np.zeros((3,3)),KK(3,1,2,y1,y2,y4),KK(3,2,1,y1,y2,y4),np.zeros((3,3)),np.transpose(Q3) * Q3]]))
    Ka1 = Ka1 + Kaa1
    Ka2 = Ka2 + Kaa2
    return ga1,ga2,Ka1,Ka2
    
    return g,K,Cell,Energy