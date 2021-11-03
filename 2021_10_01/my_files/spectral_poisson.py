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