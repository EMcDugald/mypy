import numpy as np


# QR FACTORIZATIONS


def gs_classic(A):
    row_ct = np.shape(A)[0]
    col_ct = np.shape(A)[1]
    Q = np.zeros(shape=(row_ct, col_ct))
    R = np.zeros(shape=(col_ct,col_ct))
    for j in range(1,col_ct+1):
        Q[:,j-1] = np.copy(A[:,j-1])
        for i in range(1,j):
            R[i-1,j-1] = np.dot(Q[:,i-1].T,A[:,j-1])
            Q[:,j-1] = Q[:,j-1]-R[i-1,j-1]*Q[:,i-1]
        R[j-1,j-1] = np.linalg.norm(Q[:,j-1])
        Q[:,j-1] = Q[:,j-1]/R[j-1,j-1]
    return Q, R


def gs_mod(A):
    row_ct = np.shape(A)[0]
    col_ct = np.shape(A)[1]
    Q = np.zeros(shape=(row_ct, col_ct))
    R = np.zeros(shape=(col_ct,col_ct))
    for j in range(1,col_ct+1):
        Q[:,j-1] = np.copy(A[:,j-1])
        for i in range(1,j):
            R[i-1,j-1] = np.dot(Q[:,i-1].T,Q[:,j-1])
            Q[:,j-1] = Q[:,j-1]-R[i-1,j-1]*Q[:,i-1]
        R[j-1,j-1] = np.linalg.norm(Q[:,j-1])
        Q[:,j-1] = Q[:,j-1]/R[j-1,j-1]
    return Q, R


def h_ref(A):
    m,n = np.shape(A)
    Q = np.identity(m)
    for k in range(0,n):
        z = np.copy(A[k:m,k])
        v0 = [-np.sign(z[0])*np.linalg.norm(z)-z[0]]
        v = np.append(v0,-z[1:])
        v = v/np.linalg.norm(v)
        for j in range(k,n):
            A[k:m,j] = A[k:m,j] - v*(2*(np.dot(v,A[k:m,j])))
        for j in range(0,m):
            Q[k:m,j] = Q[k:m,j] - v*(2*(np.dot(v,Q[k:m,j])))
    return Q.T, A


def h_ref_complex(A):
    A = np.copy(A).astype(dtype=np.cdouble)
    m,n = np.shape(A)
    Q = np.identity(m, dtype=np.cdouble)
    for k in range(0,n):
        z = np.copy(A[k:m,k])
        v0 = [-(z[0]/np.abs(z[0]))*np.linalg.norm(z)-z[0]]
        v = np.append(v0,-z[1:])
        v = v/np.linalg.norm(v)
        for j in range(k,n):
            A[k:m,j] = A[k:m,j] - v*(2*(np.vdot(v,A[k:m,j])))
        for j in range(0,m):
            Q[k:m,j] = Q[k:m,j] - v*(2*(np.vdot(v,Q[k:m,j])))
    return Q.conj().T, A


# LINEAR SOLVERS


# solves AX=b if you first do QR
# Ax = b => QRx = b => Rx = Q^(-1)b => Rx = Q^Tb
# So, to solve Ax=b, QR factorize, then solve the system Rx = Q^b with back subs
def back_subs(R,b):
    x = np.zeros(np.shape(b))
    n = np.shape(b)[0]
    for i in range(n-1,-1,-1):
        B = b[i]
        for j in range(n-1,i,-1):
            B -= (R[i,j]*x[j])
        x[i] = B/R[i,i]
    return x


# applies row pivots and elimantion matrices to A
# applies the same to b
# returns A and b once A is upper triangular
# can pass the result into a back substitution solver
def gauss_pp(A,b):
    col_ct = np.shape(A)[0]
    for i in range(1,col_ct+1):
        pi = get_permute(A[:,i-1],i-1)
        b = np.matmul(pi,b)
        pai = np.matmul(pi,np.copy(A))
        li = get_row_op(pai[:,i-1],i)
        b = np.matmul(li,b)
        A = np.matmul(li,pai)
    return A, b


def get_permute(v,col_num):
    p = np.identity(len(v))
    ind_of_max = np.argmax(np.abs(v[col_num:]))+col_num
    p[[ind_of_max,col_num]] = p[[col_num,ind_of_max]]
    return p


def get_row_op(v,col_num):
    l = np.identity(len(v))
    for j in range(col_num,len(v)):
        l[j,col_num-1] = -v[j]/v[col_num-1]
    return l


def solve_gpp(A,b):
    U, b = gauss_pp(A,b)
    x = back_subs(U,b)
    return x


def mat_inv(A):
    col_ct = np.shape(A)[0]
    A_inv = np.zeros(shape=(col_ct, col_ct))
    for j in range(0,col_ct):
        ej = np.eye(1, col_ct, j).T
        A_inv[0:,j] = solve_gpp(np.copy(A),ej).T
    return A_inv


# Complex analogues


def back_subs_c(R,b):
    x = np.zeros(np.shape(b),dtype=np.cdouble)
    n = np.shape(b)[0]
    for i in range(n-1,-1,-1):
        B = b[i].astype(np.cdouble)
        for j in range(n-1,i,-1):
            B -= (R[i,j]*x[j])
        x[i] = B/R[i,i]
    return x


def gauss_pp_c(A,b):
    col_ct = np.shape(A)[0]
    for i in range(1,col_ct+1):
        pi = get_permute_c(A[:,i-1],i-1)
        b = np.matmul(pi,b)
        pai = np.matmul(pi,np.copy(A))
        li = get_row_op_c(pai[:,i-1],i)
        b = np.matmul(li,b)
        A = np.matmul(li,pai)
    return A, b


def get_permute_c(v,col_num):
    p = np.identity(len(v),dtype=np.cdouble)
    ind_of_max = np.argmax(np.abs(v[col_num:]))+col_num
    p[[ind_of_max,col_num]] = p[[col_num,ind_of_max]]
    return p


def get_row_op_c(v,col_num):
    l = np.identity(len(v),dtype=np.cdouble)
    for j in range(col_num,len(v)):
        l[j,col_num-1] = -v[j]/v[col_num-1]
    return l


def solve_gpp_c(A,b):
    U, b = gauss_pp_c(A,b)
    x = back_subs_c(U,b)
    return x


def mat_inv_c(A):
    col_ct = np.shape(A)[0]
    A_inv = np.zeros(shape=(col_ct, col_ct),dtype=np.cdouble)
    for j in range(0,col_ct):
        ej = np.eye(1, col_ct, j).conj().T
        A_inv[0:,j] = solve_gpp_c(np.copy(A),ej).conj().T
    return A_inv


# EIGENVALUE/EIGENVECTOR SOLVERS


def pow_it(A,x0,n):
    xn = x0
    for i in range(1,n+1):
        xn = np.matmul(A,xn)
        xn = xn/np.linalg.norm(xn)
    return xn


def rayleigh_q(A,v):
    top = np.matmul(np.matmul(A,v).T,v)
    bottom = np.dot(v.T,v)
    return top/bottom


def rayleigh_it(A,x0,its,mu0):
    xn = x0
    mun = mu0
    I = np.identity(np.shape(A)[0])
    x = [x0]
    for i in range(1,its+1):
        xn = solve_gpp(A-mun*I,xn)
        xn = xn/np.linalg.norm(xn)
        mun = rayleigh_q(A,xn)
        x.append(xn)
    return x


def QR_it(A,its):
    Ak = np.copy(A)
    for k in range(1,its+1):
        Qk, Rk = h_ref(Ak)
        Ak = np.matmul(Rk,Qk)
    return Ak


def QR_it_shift(A,its):
    Ak = np.copy(A)
    I = np.identity(np.shape(Ak)[0])
    for k in range(1,its+1):
        Qk, Rk = h_ref(Ak-I)
        Ak = np.matmul(Rk,Qk)+I
    return Ak


def QR_it_shift_mu(A,its):
    Ak = np.copy(A).astype(dtype=np.cdouble)
    I = np.identity(np.shape(Ak)[0]).astype(dtype=np.cdouble)
    for k in range(1,its+1):
        mu = (1+1j)/2
        Qk, Rk = h_ref_complex(Ak-mu*I)
        Ak = np.matmul(Rk,Qk)+mu*I
    return Ak


# NONLINEAR SOLVERS



# when i used this initially, i defined grad and f as lambda functions...
def newt_raph_2d(x,y,grad,f):
    g = grad(x,y)
    f = f(x,y)
    return np.array([[x,y]]).T - \
    np.matmul(mat_inv_c(g),f)


def newt_raph_it_2d(max_its,init,tol,gradf,f):
    k=0
    while k <= max_its:
        x = init[0][0]
        y = init[0][1]
        if np.all(np.abs(f(x,y))<tol):
            print("iterations stopped at k=",k)
            print("final value=",init)
            print("f(x,y)=",f(x,y))
            return np.array([[x,y]])
        else:
            init = newt_raph_2d(x,y,gradf,f).T
            k +=1
    print("max iterations not reached")
    print("final point",init)

#example usage
# x0 = -1. + 1.j
# y0 = 0. + 0.j
# init = np.array([[x0,y0]])
#
# f = lambda x, y: \
#     np.array([[y-x**2,x*y-1]]).T
#
# grad = lambda x, y: \
#     np.array([[-2*x,1],[y,x]]).T
#
# newt_raph_it(20,init,1e-8,grad,f)

#bisection


def bisect_1d(f,xl,xr,tol,max_its):
    for i in range(0,max_its):
        if f(xl)*f(xr) < 0:
            x0 = (xl+xr)/2
            if np.abs(f(x0))<tol:
                print("*******************")
                print("its stopped at k=",i)
                print(x0)
                print(f(x0))
                return x0
            elif f(xl)*f(x0) < 0:
                xl = xl
                xr = x0
            elif f(xr)*f(x0) < 0:
                xl = x0
                xr = xr
            else:
                print("found nothing")
                return None
    print("max its reached, tolerance not achieved")
    print(x0)
    print(f(x0))



