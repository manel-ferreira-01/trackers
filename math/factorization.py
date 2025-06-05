import numpy as np 

def marques_factorization(obs_mat: np.array):
    """
        obs_mat: array (2*num_frame, num_features)

    """
    Xs = obs_mat[0::2,:]
    Ys = obs_mat[1::2,:]

    #center obs_mat
    tvecs = np.array([np.mean(Xs, axis=1), np.mean(Ys, axis=1)]).T
    Xs = Xs - np.array([np.mean(Xs, axis=1)]).T
    Ys = Ys - np.array([np.mean(Ys, axis=1)]).T

    obs_mat = np.vstack( (Xs, Ys) )

    U, S, Vt = np.linalg.svd(obs_mat ,full_matrices=True)
    V = Vt.T

    M_hat = U[:, 0:3] * S[0:3]**(1/2)
    S_hat = S[0:3]**(1/2) * V [: , 0:3]

    num_frame = obs_mat.shape[0] // 2

    A = []
    b = []

    for f in range(0, obs_mat.shape[0]//2):
        # i QQ^t i = 1
        A.append( constraint(M_hat[f, :], M_hat[f, :]) )
        b.append(1)

        # j QQ^t j = 1
        A.append( constraint(M_hat[num_frame+f,:], M_hat[num_frame+f,:]) )
        b.append(1)

        # i QQ^t j = 0 -> orthogonal
        A.append( constraint(M_hat[f,:], M_hat[num_frame+f, :]) )
        b.append(0)

    A = np.array(A)
    #b = np.array( [b] ).T

    mat = np.kron(  np.eye(num_frame) , np.array([[-1,-1,0]]).T)

    A = np.hstack( (A, mat) )

    #solve the svd Ax = 0
    _,_, av = np.linalg.svd(A)
    l = av[-1, :]
    l = l / l[6]
    alphas = l[6:]

    # from x build the Q matrix -> Q = LL^t
    Q = np.array( [ [l[0], l[1], l[2]], [l[1], l[3], l[4]], [l[2], l[4], l[5]] ] )
    L = np.linalg.cholesky(Q)

    M = M_hat @ L
    S = np.linalg.inv(L) @ ( S_hat.T )
    
    return M, S, tvecs, alphas

def constraint(m1,m2):
    return np.array([ m1[0]*m2[0], m1[0]*m2[1] + m1[1]*m2[0], m1[0]*m2[2] + m1[2]*m2[0], \
                    m1[1]*m2[1], m1[1]*m2[2] + m1[2]*m2[1], m1[2]*m2[2]])
