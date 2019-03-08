import scipy.sparse
from networkx.exception import NetworkXError

def pagerank_scipy(G,alpha=0.85,max_iter=1000,tol=1.0e-10,nodelist=None, personalised = False, indxs=[]):
    M=scipy.sparse.csr_matrix(G)
    (n,m)=M.shape # should be square
    S=scipy.array(M.sum(axis=1)).flatten()
    index=scipy.where(S!=0)[0]
    for i in index:
        M[i,:]*=1.0/S[i]
    x=scipy.ones((n))/n  # initial guess
    x_per = scipy.zeros((n));
    for i in indxs:
        x_per[i] = 1/len(indxs);
    dangle=scipy.array(scipy.where(M.sum(axis=1)==0,1.0/n,0)).flatten()
    for i in range(max_iter):
        xlast=x
        if personalised:
            x = alpha * (x * M + scipy.dot(dangle, xlast)) + (1 - alpha) * x_per
        else:
            x=alpha*(x*M+scipy.dot(dangle,xlast))+(1-alpha)*xlast.sum()/n
        # check convergence, l1 norm
        err=scipy.absolute(x-xlast).sum()
        if err < n*tol:
            return x
    raise NetworkXError("pagerank_scipy : power iteration failed to converge in %d iterations."%(i+1))
