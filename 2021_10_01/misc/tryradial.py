import utils
import numpy as np
import mymathradial as myrad

rmax = 0.5
rmin = 0.1

rcx = 0.3
rcy = -0.1

ndim=2**9+1
halfsize = 1.5

print(ndim)

[func,exlap] = myrad.radial(ndim,rcx,rcy,rmax,rmin,halfsize)


utils.ndwrite(exlap,"pyth_ex_lap.d")
utils.ndwrite(func,"pyth_rad_func.d")
