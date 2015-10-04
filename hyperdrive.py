from scipy.integrate import odeint
import numpy as np
from numpy.linalg import norm
import csv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from itertools import repeat

# Lengths in AU, times in days, masses scaled to saturn (S_m)
#
# Most quantities are [Body]_[property], S being Saturn, T being Titan,
# H being Hyperion.

# 6.67408E-11 from m^3/(kg*s^2) to AU^3/(Mass of Saturn * day^2) gives:
G = -8.46E-8

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

T_v_0 = [3.058426568747700E-03,
         9.550316106811974E-04,
         -7.900305243565329E-04]

H_v_0 = [1.796218227601083E-03,
         -2.119021187924069E-03,
         8.963067229904581E-04]

# Initials used by Sinclair et al.
# T_r_0 = [-0.0075533871, 0.0025250254, -0.0000462204]
# T_v_0 = [-0.0010017342, -0.0031443009, 0.0000059503]
# H_r_0 = [-0.0006436995, 0.0099145485, 0.0000357506]
# H_v_0 = [-0.0029182723, 0.0000521415, -0.0000356145]

y0 = T_r_0 + H_r_0 + T_v_0 + H_v_0

def ecc(pos):
    """
    Calculate the eccentricity of an orbit given an array of positions.
    Array must span complete stable orbit for accurate calculation
    of apses
    """
    assert isinstance(pos,np.ndarray)
    dists = norm(pos, axis=1)
    apo = np.amax(dists)
    peri = np.amin(dists)
    e = (apo - peri)/(apo + peri)
    return e

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



def f(y, t0):
    """Vector of Titan's velocity, Hyperion's velocity, T's acc, H's acc"""
    T_r = y[0:3]
    T_v = y[6:9]
    H_r = y[3:6]
    H_v = y[9:12]
    HT_sep = H_r - T_r
    T_a = G * (S_m * T_r) / norm(T_r)**3
    H_a = G * ((S_m * H_r)/norm(H_r)**3 + \
          (T_m * HT_sep)/norm(HT_sep)**3)
    vec = np.concatenate((T_v, H_v, T_a, H_a))
    return vec

# Initial and final times and timestep
t_i = 0
t_f = 160
dt = 0.001
t = np.arange(t_i, t_f, dt)

# Perdorm the integration and assign views for each quantity to dict rr.
r = odeint(f, y0, t)
quants = ('T_r', 'H_r', 'T_v', 'H_v')
longquants = ('T_x', 'T_y', 'T_z', 
              'H_x', 'H_y', 'H_z',
              'T_Vx', 'T_Vy', 'T_Vz'
              'H_Vx', 'H_Vy', 'H_Vz')
rr = dict(**{quants[i]:r[:,i*3:i*3+3] for i in range(0,len(quants))},
    **{longquants[i]:r[:,i:i+1] for i in range(0,len(longquants))})

# Array of separations from H to T
sep = norm(rr['H_r']-rr['T_r'], axis=1)


csvhead = ",".join(longquants)
np.savetxt('output.csv', r, fmt='%.6e', delimiter=',', header=csvhead)

fig = plt.figure(figsize=(8,8), facecolor='white')
fig.set_tight_layout(True)
grid = gs.GridSpec(3, 3)

orbits = plt.subplot(grid[0:2, 0:2])
orbits.set_title('Path of simulated orbits')
orbits.plot(rr['T_x'], rr['T_y'], rr['H_x'], rr['H_y'])
orbits.plot(0,0, 'xr')
orbits.axis([-.02, .02, -.02, .02])
orbits.legend(('Titan', 'Hyperion'))

seps = plt.subplot(grid[2, :])
seps.set_title('Magnitude of separation between Titan and Hyperion')
seps.plot(t, sep)
seps.text(5, np.amax(sep), 'max = {:f} AU'.format(np.amax(sep)))
seps.text(5, np.amin(sep), 'min = {:f} AU'.format(np.amin(sep)))

info = plt.subplot(grid[0:-1, -1])
info.set_title('Info')
info.set_xticks([])
info.set_yticks([])
tab = info.table(rowLabels=['Hyperion e', 'Titan e'],
           cellText=[[1], [2]],
           loc='upper right',
           colWidths=[0.5]*2)
#Whoever wrote the Table class hates legibility. Let's increase the row height
for c in tab.properties()['child_artists']:
    c.set_height(c.get_height()*2)

plt.show()