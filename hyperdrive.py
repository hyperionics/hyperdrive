from scipy.integrate import odeint
import numpy as np
from numpy.linalg import norm
import csv

# Lengths in AU, times in days, masses scaled to Mass_saturn (S_m)
#
# Most quantities are [Body]_[property], S being Saturn, T being Titan,
# H being Hyperion.

# 6.67408E-11 from m^3/(kg*s^2) to AU^3/(Mass of Saturn * day^2) gives:
G = 8.46E-8

# Masses
S_m = 1
T_m = 2.367E-4
H_m = 9.8E-9

#Initial values for the positions and velocities
T_r_0 = [2.806386833950917E-03,
         -6.729009690324418E-03,
         3.202694551398282E-03]

T_v_0 = [3.058426568747700E-03,
         9.550316106811974E-04,
         -7.900305243565329E-04]

H_r_0 = [-7.240723412782416E-03,
         -5.692266853965866E-03,
         3.722613581954420E-03]

H_v_0 = [1.796218227601083E-03,
         -2.119021187924069E-03,
         8.963067229904581E-04]

def f(y, t0):
    """Vector of Titan's velocity, Hyperion's velocity, T's acc, H's acc"""
    T_r = y[0:3]
    T_v = y[6:9]
    H_r = y[3:6]
    H_v = y[9:12]
    HT_sep = H_r - T_r
    T_a = (G / T_m) * (S_m * T_r) / norm(T_r)**3
    H_a = (G / H_m) * ((S_m * H_r)/norm(H_r)**3 + \
          (T_m * HT_sep)/norm(HT_sep)**3)
    vec = np.concatenate((T_v, H_v, T_a, H_a))
    return vec

# Initial and final times and timestep
t_i = 0
t_f = 16
dt = 0.001


y0 = T_r_0 + H_r_0 + T_v_0 + H_v_0
results = odeint(f, y0, np.arange(t_i, t_f, dt))

np.savetxt('output.csv', results, fmt='%.6e', delimiter=',', header="Tx, Ty, Tz, TVx, TVy, TVz, Hx, Hy, Hz, HVx, HVy, HVz")
plt.plot(r[:,0], r[:,1], r[:,6], r[:,7])
plt.axis([-.01, .01, -.01, .01])
plt.show()