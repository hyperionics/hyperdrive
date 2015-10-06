from scipy.integrate import odeint
import numpy as np
from numpy.linalg import norm
import csv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from math import sqrt, sin, cos, asin, acos, copysign
from functools import partial

# Lengths in AU, times in days, masses scaled to saturn (S_m)
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

H_r_0 = [-7.240723412782416E-03,
         -5.692266853965866E-03,
         3.722613581954420E-03]

T_theta_0 = [0,0,0]

H_theta_0 = [0,0,0]

T_v_0 = [3.058426568747700E-03,
         9.550316106811974E-04,
         -7.900305243565329E-04]

H_v_0 = [1.796218227601083E-03,
         -2.119021187924069E-03,
         8.963067229904581E-04]

T_omega_0 = [0,0,0]

H_omega_0 = [0,0,0]

# Initials used by Sinclair et al.
# T_r_0 = [-0.0075533871, 0.0025250254, -0.0000462204]
# T_v_0 = [-0.0010017342, -0.0031443009, 0.0000059503]
# H_r_0 = [-0.0006436995, 0.0099145485, 0.0000357506]
# H_v_0 = [-0.0029182723, 0.0000521415, -0.0000356145]

y0 = T_r_0 + H_r_0 + T_theta_0 + H_theta_0 + \
     T_v_0 + H_v_0 + T_omega_0 + H_omega_0

def period(pos, times):
    """
    Find the orbital period given an array of positions in MORE THAN ONE
    complete stable orbit and an array of corresponding times
    """
    x = pos[:,0]
    assert isinstance(pos,np.ndarray)
    currentsign = np.sign(x[0])
    count = crossings = 0
    crosstimes = []
    while crossings < 3:
        if np.sign(x[count]) == currentsign:
            count += 1
        else:
            crossings += 1
            currentsign = np.sign(x[count])
            count += 1
            crosstimes.append(times[count])
    assert len(crosstimes) == 3
    return crosstimes[2] - crosstimes[1]

def kepler(pos, vel, m):
    """
    Returns a dict of arrays of orbital elements (and other stuff) from arrays 
    of position, velocity and mass
    """
    R = norm(pos, axis=1)
    V = norm(vel, axis=1)
    h = np.cross(pos, vel)
    mu = G*(S_m+m)
    # Semi-Major Axis
    sma = 1/(2/R - V**2/mu)
    # Eccentricity
    ecc = np.sqrt(1-norm(h, axis=1)**2/(sma*mu))
    # Inclination
    inc = np.arccos(h[:,2]/norm(h, axis=1))
    # Longitude of Ascending Node
    lan = np.arcsin(np.copysign(h[:,0],h[:,2])/(norm(h, axis=1)*np.sin(inc)))
    # True Anomaly
    tra = sma*norm(vel, axis=1)*(1-ecc**2)/(norm(h, axis=1)*ecc)
    # Argument of Periapsis    
    arg = np.arcsin(pos[:,2]/(norm(pos, axis=1)*np.sin(inc))) - tra
    # Mean motion
    mm = np.sqrt(mu/sma**3)/(2*np.pi)
    dt = [(i, float) for i in ['sma', 'ecc', 'inc',
                               'lan', 'tra', 'arg',
                               'mm']]
    return np.fromiter(zip(sma, ecc, inc, lan, tra, arg, mm), dt)

def f(y, t0):
    """Vector of Titan's velocity, Hyperion's velocity, T's acc, H's acc"""
    [T_r, H_r, T_theta, H_theta, T_v, H_v, T_omega, H_omega] = \
    [y[i:i+3] for i in range(0, len(y), 3)]
    HT_sep = H_r - T_r
    T_a = -G * (S_m * T_r) / norm(T_r)**3
    H_a = -G * ((S_m * H_r)/norm(H_r)**3 + \
          (T_m * HT_sep)/norm(HT_sep)**3)
    T_alpha = [0,0,0]
    H_alpha = [0,0,0]
    vec = np.concatenate((T_v, H_v, T_omega, H_omega,
                          T_a, H_a, T_alpha, H_alpha))
    return vec

# Initial and final times and timestep
t_i = 0
t_f = 160
dt = 0.001
t = np.arange(t_i, t_f, dt)

# Perdorm the integration and assign views for each quantity to dict rr.
r = odeint(f, y0, t)
quants = ('T_r', 'H_r', 'T_theta', 'H_theta',
          'T_v', 'H_v', 'T_omega', 'H_omega')
longquants = ('T_x', 'T_y', 'T_z', 
              'H_x', 'H_y', 'H_z',
              'T_T1', 'T_T2', 'T_T3',
              'H_T1', 'H_T2', 'H_T3',
              'T_Vx', 'T_Vy', 'T_Vz',
              'H_Vx', 'H_Vy', 'H_Vz',
              'T_O1', 'T_O2', 'T_O3',
              'H_O1', 'H_O2', 'H_O3')
rr = dict(**{quants[i]:r[:,i*3:i*3+3] for i in range(0,len(quants))},
    **{longquants[i]:r[:,i:i+1] for i in range(0,len(longquants))})

# Array of separations from H to T
sep = norm(rr['H_r']-rr['T_r'], axis=1)

H_elem = kepler(rr['H_r'], rr['H_v'], H_m)
T_elem = kepler(rr['T_r'], rr['T_v'], T_m)

H_elem_0 = H_elem[0]
T_elem_0 = T_elem[0]

csvhead = ",".join(longquants)
np.savetxt('output.csv', r, fmt='%.6e', delimiter=',', header=csvhead)

fig = plt.figure(figsize=(8,8), facecolor='white')
fig.set_tight_layout(True)
grid = gs.GridSpec(3, 3)

orbits = plt.subplot(grid[0:2, 0:2])
orbits.set_title('Path of simulated orbits')
orbits.plot(rr['T_x'], rr['T_y'], rr['H_x'], rr['H_y'])
orbits.plot(0,0, 'xr')
orbits.axis([-.015, .015, -.015, .015])
orbits.legend(('Titan', 'Hyperion'))

seps = plt.subplot(grid[2, :])
seps.set_title('Magnitude of separation between Titan and Hyperion')
seps.plot(t, sep)

info = plt.subplot(grid[0:-1, -1])
info.set_title('Info')
info.axis('off')
labels = [
    'Hyp init e',
    'Hyp mean e',
    'Tit init e',
    'Tit mean e',
    'Init n/n',
    'Mean n/n',
    'Max separation',
    'Min separation'
    ]
text = [[i] for i in map(partial(round, ndigits=3), [
    H_elem_0['ecc'],
    np.mean(H_elem['ecc']),
    T_elem_0['ecc'],
    np.mean(T_elem['ecc']),
    H_elem_0['mm']/T_elem_0['mm'],
    np.mean(H_elem['mm']/T_elem['mm'])])
    ] + \
    [[i] for i in map(partial(round, ndigits=5), [
    np.amax(sep),
    np.amin(sep)])
    ]
tab = info.table(rowLabels=labels,
           cellText=text,
           loc='upper right',
           colWidths=[0.5]*2)
#Whoever wrote the Table class hates legibility. Let's increase the row height
for c in tab.properties()['child_artists']:
    c.set_height(c.get_height()*2)

plt.show()