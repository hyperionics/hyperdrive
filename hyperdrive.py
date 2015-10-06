from scipy.integrate import odeint
import numpy as np
from numpy.linalg import norm
import csv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from math import sqrt, sin, cos, tan, asin, acos, atan, copysign
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

# Dimensionless moi
H_A = 0.314
H_B = 0.474
H_C = 0.542

T_A = T_B = T_C = 0.3414

#Initial values for the positions and velocities, from HORIZONS on 2005-09-25
T_r_0 = [-4.088407843090480E-03,
         -6.135746299499617E-03,
         3.570710328903993E-03]

H_r_0 = [-9.556008109223760E-03,
         -6.330869216667536E-04,
         1.293277241526503E-03]

T_theta_0 = [0,0,0]
H_theta_0 = [0,0,0]

# From Harbison, Cassini flyby on 2005-09-25
H_euler_0 = [2.989, 1.685, 1.641]
H_wisdom_0 = [0,0,0]

T_v_0 = [2.811008677848850E-03,
         -1.470816650319267E-03,
         4.806729268010478E-04]

H_v_0 = [6.768811686476851E-04,
         -2.639837861571281E-03,
         1.253587766343593E-03]

# Titan's rotation is orbit-synchronous, so it only rotates around its
# principal axis. Since its axial tilt is zero, this is always normal
# to the orbital plane.
T_per = 1.594772769327679E+01 #From HORIZONS on 2005-09-25
T_omega_0 = [0,0,2*np.pi/T_per]

# From Harbison, Cassini flyby on 2005-09-25
H_omega_0_mag = 72*np.pi/180
H_omega_0 = [i * H_omega_0_mag for i in [0.902, 0.133, 0.411]]

# Initials used by Sinclair et al.
# T_r_0 = [-0.0075533871, 0.0025250254, -0.0000462204]
# T_v_0 = [-0.0010017342, -0.0031443009, 0.0000059503]
# H_r_0 = [-0.0006436995, 0.0099145485, 0.0000357506]
# H_v_0 = [-0.0029182723, 0.0000521415, -0.0000356145]

y0 = T_r_0 + H_r_0 + T_theta_0 + H_theta_0 + H_euler_0 + H_wisdom_0 + \
     T_v_0 + H_v_0 + T_omega_0 + H_omega_0

def kepler(pos, vel, m):
    """
    Returns a dict of arrays of orbital elements (and other stuff) from arrays 
    of position, velocity and mass. Angles are in radians.
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
 
def wis(euler):
    """Transform from Euler angles to Wisdom angles"""
    assert len(euler) == 3
    e_theta, e_phi, e_psi = euler[0:3]
    w_theta = atan((cos(e_theta)*sin(e_psi) + sin(e_theta)*cos(e_phi)*cos(e_psi))/\
        (cos(e_theta)*cos(e_phi)*cos(e_psi) - sin(e_theta)*sin(e_psi)))
    w_phi = atan(sin(e_phi)*cos(e_psi))
    w_psi = atan(-sin(e_phi)*sin(e_psi)/cos(e_phi))
    return [w_theta, w_phi, w_psi]

def eul(wisdom):
    """Transform from Wisdom angles to Euler angles"""
    assert len(wisdom) == 3
    w_theta, w_phi, w_psi = wisdom[0:3]
    e_theta = atan((cos(w_theta)*sin(w_psi) + sin(w_theta)*cos(w_phi)*cos(w_psi))/\
        (cos(w_theta)*cos(w_phi)*cos(w_psi) - sin(w_theta)*sin(w_psi)))
    e_phi = atan(1/(cos(w_phi)*cos(w_psi)))
    e_psi = atan(-cos(w_phi)*sin(w_psi)/sin(w_phi))
    return [e_theta, e_phi, e_psi]

def ecc(pos, vel, m):
    """Calculate the eccentricity vector from the state vector and mass"""
    return np.cross(vel, np.cross(pos, vel))/(G*(S_m+m)) - pos/norm(pos)

def dircos(euler, wisdom, anom):
    """Calculate the directional cosines from a body towards Saturn"""
    assert len(euler) == len(wisdom) == 3
    if norm(euler) == 0:
        theta, phi, psi = wisdom[0:3]
        alpha = cos(theta - anom)*cos(psi) - sin(theta - anom)*sin(phi)*sin(psi)
        beta = sin(theta - anom)*cos(phi)
        gamma = cos(theta - anom)*sin(psi) + sin(theta - anom)*sin(phi)*cos(psi)
    else:
        theta, phi, psi = euler[0:3]
        alpha = cos(theta - anom)*cos(psi) - sin(theta - anom)*cos(phi)*sin(psi)
        beta = cos(theta - anom)*sin(-psi) - sin(theta - anom)*cos(phi)*cos(psi)
        gamma = sin(theta-anom)*sin(phi)
    return [alpha, beta, gamma]

def eulerderivs(euler, omega):
    """Calculate the time-derivatives of the euler angles due to ang. vel."""
    assert len(euler) == 3 and len(omega) == 3
    theta, phi, psi = euler[0:3]
    Dtheta = (omega[0]*sin(psi) + omega[1]*cos(psi))/sin(phi)
    Dphi = omega[0]*cos(psi) - omega[1]*sin(psi)
    Dpsi = omega[2] - Dtheta*cos(phi)
    return [Dtheta, Dphi, Dpsi]

def f(y, t0):
    """Vector of Titan's velocity, Hyperion's velocity, T's acc, H's acc"""
    [T_r, H_r, T_theta, H_theta, H_euler, H_wisdom, T_v, H_v, T_omega, H_omega] = \
    [y[i:i+3] for i in range(0, len(y), 3)]

    if norm(H_wisdom) == 0 and abs(sin(H_euler[1])) <= 10E-2:
        H_wisdom = wis(H_euler)
        H_euler = [0,0,0]
        # print(t0, H_wisdom, "NOW USING WISDOM")

    if norm(H_euler) == 0 and abs(cos(H_euler[1])) <= 10E-2:
        H_euler = eul(H_wisdom)
        H_wisdom = [0,0,0]
        # print(t0, H_wisdom, "NOW USING EULER")
        
    H_r_ = norm(H_r)
    HT_sep = H_r - T_r

    T_a = -G * (S_m * T_r) / norm(T_r)**3
    H_a = -G * ((S_m * H_r)/norm(H_r)**3 + \
          (T_m * HT_sep)/norm(HT_sep)**3)

    H_ecc = ecc(H_r, H_v, H_m)
    T_ecc = ecc(T_r, T_v, T_m)

    H_anom = acos(np.dot(H_ecc, H_r)/(norm(H_ecc)*norm(H_r)))
    if np.dot(H_r, H_v) < 0: H_anom = 2*np.pi - H_anom
    T_anom = acos(np.dot(T_ecc, T_r)/(norm(T_ecc)*norm(T_r)))
    if np.dot(T_r, T_v) < 0: T_anom = 2*np.pi - T_anom

    H_dircos = dircos(H_euler, H_wisdom, H_anom)

    if norm(H_wisdom) == 0:
        H_D_euler = eulerderivs(H_euler, H_omega)
    else:
        H_D_euler = [0,0,0]

    if norm(H_euler) == 0:
        H_D_wisdom = eulerderivs(H_wisdom, H_omega)
    else:
        H_D_wisdom = [0,0,0]

    # Titan's MoIs are all equal, so by Harbison it undergoes no ang. acc.
    T_alpha = [0, 0, 0]
    H_alpha = [((H_B-H_C)/H_A)*(H_omega[1]*H_omega[2] - \
                   (3/H_r_**3)*H_dircos[1]*H_dircos[2]),
               ((H_C-H_A)/H_B)*(H_omega[0]*H_omega[2] - \
                   (3/H_r_**3)*H_dircos[0]*H_dircos[2]),
               ((H_A-H_B)/H_C)*(H_omega[0]*H_omega[1] - \
                   (3/H_r_**3)*H_dircos[0]*H_dircos[1])]
    # H_alpha = [0,0,0]

    vec = np.concatenate((T_v, H_v, T_omega, H_omega, H_D_euler, H_D_wisdom,
                          T_a, H_a, T_alpha, H_alpha))
    return vec

# Initial and final times and timestep
t_i = 0
t_f = 1
dt = 0.001
t = np.arange(t_i, t_f, dt)

# Perform the integration and assign views for each quantity to dict rr.
r = odeint(f, y0, t)
quants = ('T_r', 'H_r', 'T_theta', 'H_theta', 'H_euler', 'H_wisdom',
          'T_v', 'H_v', 'T_omega', 'H_omega')
longquants = ('T_x', 'T_y', 'T_z', 
              'H_x', 'H_y', 'H_z',
              'T_T1', 'T_T2', 'T_T3',
              'H_T1', 'H_T2', 'H_T3',
              'H_E1', 'H_E2', 'H_E3',
              'H_W1', 'H_W2', 'H_W3',
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

fig = plt.figure(figsize=(8, 8), facecolor='white')
fig.set_tight_layout(True)
grid = gs.GridSpec(3, 3)

orbits = plt.subplot(grid[0:2, 0:2])
orbits.set_title('Path of simulated orbits')
orbits.plot(rr['T_x'], rr['T_y'], rr['H_x'], rr['H_y'])
orbits.plot(0,0, 'xr')
orbits.axis([-.015, .015, -.015, .015])
orbits.legend(('Titan', 'Hyperion'))
seps = plt.subplot(grid[2, 0:3])
seps.set_title('Magnitude of separation between Titan and Hyperion')
seps.set_xlabel('days')
seps.set_ylabel('AU')
seps.plot(t, sep)

info = plt.subplot(grid[0:2, -1])
info.set_title('Info')
info.axis('off')
labels = [
    'Hyp init e',
    'Hyp mean e',
    'Tit init e',
    'Tit mean e',
    'Init n/n',
    'Mean n/n',
    'Hyp mean incl',
    'Max separation',
    'Min separation'
    ]
text = [[i] for i in map(partial(round, ndigits=3), [
    H_elem_0['ecc'],
    np.mean(H_elem['ecc']),
    T_elem_0['ecc'],
    np.mean(T_elem['ecc']),
    H_elem_0['mm']/T_elem_0['mm'],
    np.mean(H_elem['mm']/T_elem['mm']),
    np.mean(H_elem['inc'])])
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

fig2 = plt.figure(figsize=(16, 8), facecolor='white')
grid2 = gs.GridSpec(2, 6)

titan = plt.subplot(grid2[0,:])
titan.plot(t, rr['T_T1'], t, rr['T_T2'], t, rr['T_T3'])
titan.axis([0, 1, -.1, .1])

hyperion = plt.subplot(grid2[1,:])
hyperion.plot(t, rr['H_T1'], t, rr['H_T2'], t, rr['H_T3'])
hyperion.axis([0, 1, -.1, .1])

plt.show()