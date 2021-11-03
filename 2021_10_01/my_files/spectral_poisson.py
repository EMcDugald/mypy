import numpy as np
import mymath

#solves for coefficients in fourier transform
# see https://en.wikipedia.org/wiki/Spectral_method for an example
def get_bks(fft_mat):
    k1_ind = np.shape(fft_mat)[0]
    k2_ind = np.shape(fft_mat)[1]
    U = np.zeros(shape=(k1_ind,k2_ind))
    for k1 in range(1,k1_ind):
        for k2 in range(1,k2_ind):
            U[k1,k2] = -fft_mat[k1,k2]/((k1)**2+(k2)**2)
    return U


#returns un-normalized solution from laplacian
#in_dim = inout dimensions- defines normalization
def return_u(lap,in_dim):
    norm_x1 = (in_dim[0][1]-in_dim[0][0])/np.pi
    norm_x2 = (in_dim[1][1]-in_dim[1][0])/np.pi
    # get coefficients of dft of laplacian u
    ak_mat = mymath.mydst2(np.copy(lap))
    # solve for the coefficients of u in frequency domain
    bk_mat = get_bks(np.copy(ak_mat))
    # apply the dft to get the time/space domain values of u
    u = mymath.mydst2(np.copy(bk_mat))
    return u*norm_x1*norm_x2


############# WIP ###############


def zeta(x1,x2,rmin,rmax):
    return mymath.t6hat(rmax,rmin,x1)*mymath.t6hat(rmax,rmin,x2)


def chi(x1, x2, rmin):
    t1 = np.abs(x1) < rmin
    t2 = np.abs(x2) < rmin
    t3 = np.logical_and(t1, t2)
    p = np.where(t3 == True, 0, 1)
    return p


def laplacian(dx, dy, w):
    laplacian_xy = np.zeros(w.shape)
    for y in range(w.shape[1]-1):
        laplacian_xy[:, y] = (1/dy)**2 * ( w[:, y+1] - 2*w[:, y] + w[:, y-1] )
    for x in range(w.shape[0]-1):
        laplacian_xy[x, :] = laplacian_xy[x, :] + (1/dx)**2 * ( w[x+1,:] - 2*w[x,:] + w[x-1,:] )
    return laplacian_xy


def phi(x1,x2):
    norm = np.sqrt(x1**2+x2**2)
    return np.log(norm)/(2*np.pi)


def G(u,x1,x2,rmin,rmax,dx,dy):
    w = u*zeta(x1,x2,rmin,rmax)
    return laplacian(dx,dy,w)