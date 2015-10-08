from scipy.integrate import odeint
import numpy as np
from numpy.linalg import norm
import csv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from math import sqrt, sin, cos, tan, asin, acos, atan, atan2, copysign
from functools import partial

# Lengths in AU, times in days, masses scaled to saturn (S_m)
#
# Most quantities are [Body]_[property], S being Saturn, T being Titan,
# H being Hyperion.

AU = 149597871

# 6.67408E-11 from m^3/(kg*s^2) to AU^3/(Mass of Saturn * day^2) gives:
G = 8.46E-8 * AU**3

# Masses
S_m = 1
T_m = 2.367E-4
H_m = 9.8E-9

# Radii and Js
# Titan J2 from Iess and R from Zebker
T_J2 = 31.808E-6
T_R = 2574.91
# Saturn J2 form Jacobson and R from NSSDC (Nasa)
S_J2 = 16299E-6
S_R = 60268

# Dimensionless moi
H_A = 0.314
H_B = 0.474
H_C = 0.542
H_BCA = (H_B - H_C)/H_A
H_CAB = (H_C - H_A)/H_B
H_ABC = (H_A - H_B)/H_C

T_A = T_B = T_C = 0.3414

#Initial values for the positions and velocities, from HORIZONS on 2005-09-25
T_r_0 = [-4.088407843090480E-03 *AU,
         -6.135746299499617E-03 *AU,
         3.570710328903993E-03 *AU]

H_r_0 = [-9.556008109223760E-03 *AU,
         -6.330869216667536E-04 *AU,
         1.293277241526503E-03 *AU]

T_theta_0 = [0,0,0]
H_theta_0 = [0,0,0]

# From Harbison, Cassini flyby on 2005-09-25
H_euler_0 = [2.989, 1.685, 1.641]
H_wisdom_0 = [0,0,0]

T_v_0 = [2.811008677848850E-03 *AU,
         -1.470816650319267E-03 *AU,
         4.806729268010478E-04 *AU]

H_v_0 = [6.768811686476851E-04 *AU,
         -2.639837861571281E-03 *AU,
         1.253587766343593E-03 *AU]

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

def wis(euler):
    """Transform from Euler angles to Wisdom angles"""
    assert len(euler) == 3
    e_theta, e_phi, e_psi = euler[0:3]
    w_theta = atan((cos(e_theta)*sin(e_psi) + sin(e_theta)*cos(e_phi)*cos(e_psi))/\
        (cos(e_theta)*cos(e_phi)*cos(e_psi) - sin(e_theta)*sin(e_psi)))
    w_phi = atan(sin(e_phi)*cos(e_psi))
    w_psi = atan(-sin(e_phi)*sin(e_psi)/cos(e_phi))
    return [w_theta, w_phi, w_psi]

# Use Wisdom angles instead of Euler angles!
H_wisdom_0 = wis(H_euler_0)

# Combine initial conditions into a handy vector
y0 = T_r_0 + H_r_0 + T_theta_0 + H_theta_0 + H_wisdom_0 + \
     T_v_0 + H_v_0 + T_omega_0 + H_omega_0

def ecc(pos, vel, m):
    """Calculate the eccentricity vector from the state vector and mass"""
    return np.cross(vel, np.cross(pos, vel))/(G*(S_m+m)) - pos/norm(pos)

def rowdot(v1, v2):
    return np.einsum('ij, ij->i', v1, v2)

def kepler(pos, vel, m):
    """
    Returns a dict of arrays of orbital elements (and other stuff) from arrays 
    of position, velocity and mass. Angles are in radians.
    """
    R = norm(pos, axis=1)
    V = norm(vel, axis=1)
    h = np.cross(pos, vel)
    k = [0,0,1]
    n = np.cross(k, h)
    mu = G*(S_m+m)
    Vecc = np.cross(vel, np.cross(pos, vel))/(G*(S_m+m)) - pos/np.vstack([norm(pos, axis=1)]*3).T
    # Semi-Major Axis
    sma = 1/(2/R - V**2/mu)
    # Eccentricity
    ecc = norm(Vecc, axis=1)
    # Inclination
    inc = np.arccos(h[:,2]/norm(h, axis=1))
    # Longitude of Ascending Node
    lan = np.arccos(n[:,0]/norm(n, axis=1))
    # True Anomaly
    posdotvel = rowdot(pos, vel)
    tra = np.arccos(rowdot(Vecc, pos)/(norm(Vecc, axis=1)*norm(pos, axis=1)))
     # Argument of Periapsis    
    arg = np.arccos(rowdot(n, Vecc)/(norm(n, axis=1)*norm(Vecc, axis=1)))
    # Mean motion
    mm = np.sqrt(mu/sma**3)/(2*np.pi)
    # Make all the signs right
    for i in range(0, len(R)):
        if posdotvel[i] < 0:
            tra[i] = 2*np.pi - tra[i]
        if Vecc[i, 2] < 0:
            arg[i] = 2*np.pi - arg[i]
        if n[i, 1] < 0:
            lan = 2*np.pi - lan[i]
    dt = [(i, float) for i in ['sma', 'ecc', 'inc',
                               'lan', 'tra', 'arg',
                               'mm']]
    return np.fromiter(zip(sma, ecc, inc, lan, tra, arg, mm), dt)

def dircos(wisdom, anom):
    """Calculate the directional cosines from a body towards Saturn"""
    theta, phi, psi = wisdom[0:3]
    alpha = cos(theta - anom)*cos(psi) - sin(theta - anom)*sin(phi)*sin(psi)
    beta = sin(theta - anom)*cos(phi)
    gamma = cos(theta - anom)*sin(psi) + sin(theta - anom)*sin(phi)*cos(psi)
    return [alpha, beta, gamma]

def eulerderivs(wisdom, omega):
    """Calculate the time-derivatives of the euler angles due to ang. vel."""
    assert len(wisdom) == len(omega) == 3
    theta, phi, psi = wisdom[0:3]
    Dtheta = (omega[0]*sin(psi) + omega[1]*cos(psi))/sin(phi)
    Dphi = omega[0]*cos(psi) - omega[1]*sin(psi)
    Dpsi = omega[2] - Dtheta*cos(phi)
    return [Dtheta, Dphi, Dpsi]

def flattenacc(pos, R, J2):
    r = norm(pos)
    theta = acos(pos[2]/r)
    phi = atan2(pos[1],pos[0])
    x = (1/2) * (3*cos(theta)**2-1) * sin(theta) * cos(phi) + \
        sin(theta)*cos(theta)**2
    y = (1/2) * sin(theta) * sin(phi) * (3*cos(theta)**2-1) + \
        sin(theta)*cos(theta)**2*sin(phi)
    z = (1/2) * cos(theta) * (3*cos(theta)**2-1) - \
        sin(theta)**2*cos(theta)
    return np.multiply(3*J2*G*R**2/r**4, [x, y, z])

def poinsect(a, af, f):
    through_sect = np.full((len(af),), False, dtype=bool)
    for i in range(0,len(af)):
        if f == 0:
            if af[i-1] > af[i]:
                through_sect[i] = True
        else:
            if af[i-1] < f < af[i]:
                through_sect[i] = True
    out = a[through_sect]
    return out

def f(y, t0):
    """Vector of Titan's velocity, Hyperion's velocity, T's acc, H's acc"""
    [T_r, H_r, T_theta, H_theta, H_wisdom, T_v, H_v, T_omega, H_omega] = \
    [y[i:i+3] for i in range(0, len(y), 3)]

    # for i in [T_theta, H_theta, H_wisdom]:
    #     for j in range(0,3):
    #         if np.isclose(i[j], 0): i[j] = int(0)
    #         i[j] = atan2(sin(i[j]), cos(i[j]))

    H_r_ = norm(H_r)
    HT_sep = H_r - T_r

    ST_flat = flattenacc(T_r, S_R, S_J2)
    assert acos(np.dot(ST_flat, -T_r)/(norm(ST_flat)*norm(T_r))) <= np.pi/2
    SH_flat = flattenacc(H_r, S_R, S_J2)
    assert acos(np.dot(SH_flat, -H_r)/(norm(SH_flat)*norm(H_r))) <= np.pi/2
    TH_flat = flattenacc(HT_sep, T_R, T_J2)
    assert acos(np.dot(TH_flat, -HT_sep)/(norm(TH_flat)*norm(HT_sep))) <= np.pi/2

    T_a = -G * (S_m * T_r) / norm(T_r)**3 + ST_flat
    H_a = -G * ((S_m * H_r)/norm(H_r)**3 + \
          (T_m * HT_sep)/norm(HT_sep)**3) + \
          SH_flat + TH_flat

    H_ecc = ecc(H_r, H_v, H_m)
    T_ecc = ecc(T_r, T_v, T_m)

    H_anom = acos(np.dot(H_ecc, H_r)/(norm(H_ecc)*norm(H_r)))
    if np.dot(H_r, H_v) < 0: H_anom = 2*np.pi - H_anom
    T_anom = acos(np.dot(T_ecc, T_r)/(norm(T_ecc)*norm(T_r)))
    if np.dot(T_r, T_v) < 0: T_anom = 2*np.pi - T_anom

    H_dircos = dircos(H_wisdom, H_anom)

    # H_D_euler, H_D_wisdom = eulerderivs(H_euler, H_wisdom, H_omega)
    H_D_wisdom = eulerderivs(H_wisdom, H_omega)

    # Titan's MoIs are all equal, so by Harbison it undergoes no ang. acc.
    T_alpha = [0, 0, 0]

    H_alpha = [H_BCA*(H_omega[1]*H_omega[2] - \
                   (3*G*S_m/H_r_**3)*H_dircos[1]*H_dircos[2]),
               H_CAB*(H_omega[0]*H_omega[2] - \
                   (3*G*S_m/H_r_**3)*H_dircos[0]*H_dircos[2]),
               H_ABC*(H_omega[0]*H_omega[1] - \
                   (3*G*S_m/H_r_**3)*H_dircos[0]*H_dircos[1])]

    vec = np.concatenate((T_v, H_v, T_omega, H_omega, H_D_wisdom,
                          T_a, H_a, T_alpha, H_alpha))
    return vec

# Initial and final times and timestep
t_i = 0
t_f = 640
dt = 0.001
t = np.arange(t_i, t_f, dt)

# Perform the integration and assign views for each quantity to dict rr.
r = odeint(f, y0, t)
quants = ('T_r', 'H_r', 'T_theta', 'H_theta', 'H_wisdom',
          'T_v', 'H_v', 'T_omega', 'H_omega')
longquants = ('T_x', 'T_y', 'T_z', 
              'H_x', 'H_y', 'H_z',
              'T_T1', 'T_T2', 'T_T3',
              'H_T1', 'H_T2', 'H_T3',
              'H_W1', 'H_W2', 'H_W3',
              'T_Vx', 'T_Vy', 'T_Vz',
              'H_Vx', 'H_Vy', 'H_Vz',
              'T_O1', 'T_O2', 'T_O3',
              'H_O1', 'H_O2', 'H_O3')
rr = dict(**{quants[i]:r[:,i*3:i*3+3] for i in range(0,len(quants))},
    **{longquants[i]:r[:,i:i+1] for i in range(0,len(longquants))})

for i in [rr['T_theta'], rr['H_theta'], rr['H_wisdom']]:
    for j in range(0, len(t)):
        for k in range(0, 3):
            i[j,k] = atan2(sin(i[j,k]),cos(i[j,k]))

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
plt.rcParams['axes.formatter.limits'] = [-5,5]

orbits = plt.subplot(grid[0:2, 0:2])
orbits.set_title('Path of simulated orbits')
orbits.plot(rr['T_x'], rr['T_y'], rr['H_x'], rr['H_y'])
orbits.plot(0,0, 'xr')
orbits.axis([-2E6, 2E6, -2E6, 2E6])
orbits.legend(('Titan', 'Hyperion'))

seps = plt.subplot(grid[2, 0:3])
seps.set_title('Magnitude of separation between Titan and Hyperion')
seps.set_xlabel('days')
seps.set_ylabel('km')
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

figps = plt.figure(figsize=(12, 3), facecolor='white')
gridps = gs.GridSpec(1, 4)
figps.set_tight_layout(True)

hyp_o1_t1 = plt.subplot(gridps[:,0])
hyp_o1_t1.scatter(poinsect(rr['H_T1'], H_elem['tra'], 0),
                  poinsect(rr['H_O1'], H_elem['tra'], 0),
                  marker='.')
hyp_o1_t1.set_xlabel('Theta 1')
hyp_o1_t1.set_ylabel('d(Theta 1)/dt')
hyp_o1_t1.axis([-4, 4, -1.5, 1.5])


hyp_o2_t2 = plt.subplot(gridps[:,1])
hyp_o2_t2.scatter(poinsect(rr['H_T2'], H_elem['tra'], 0),
                  poinsect(rr['H_O2'], H_elem['tra'], 0),
                  marker='.')
hyp_o2_t2.set_xlabel('Theta 2')
hyp_o2_t2.set_ylabel('d(Theta 2)/dt')
hyp_o2_t2.axis([-4, 4, -1.5, 1.5])


hyp_o3_t3 = plt.subplot(gridps[:,2])
hyp_o3_t3.scatter(poinsect(rr['H_T3'], H_elem['tra'], 0),
                  poinsect(rr['H_O3'], H_elem['tra'], 0),
                  marker='.')
hyp_o3_t3.set_xlabel('Theta 3')
hyp_o3_t3.set_ylabel('d(Theta 3)/dt')
hyp_o3_t3.axis([-4, 4, -1.5, 1.5])


hyp_o_t = plt.subplot(gridps[:,3])
hyp_o_t.scatter(poinsect(norm(rr['H_theta'], axis=1), H_elem['tra'], 0),
                poinsect(norm(rr['H_omega'], axis=1), H_elem['tra'], 0),
                marker='.')
hyp_o_t.set_xlabel('|Theta|')
hyp_o_t.set_ylabel('d|Theta|/dt')

# peri = plt.subplot(grid[3,:])
# peri.plot(t, (T_elem['lan']+T_elem['arg']) - (H_elem['lan']+H_elem['arg']))

fig2 = plt.figure(figsize=(12, 6), facecolor='white')
grid2 = gs.GridSpec(2, 6)

titan = plt.subplot(grid2[0,:])
titan.plot(t, rr['T_O1'], t, rr['T_O2'], t, rr['T_O3'])
titan.axis([0, t[-1], -np.pi, np.pi])

hyperion = plt.subplot(grid2[1,:])
hyperion.plot(t, rr['H_O1'], t, rr['H_O2'], t, rr['H_O3'])
hyperion.axis([0, t[-1], -np.pi, np.pi])
for i in range(0, len(t)):      
    if H_elem['tra'][i-1] > H_elem['tra'][i]: hyperion.axvline(t[i])
    if T_elem['tra'][i-1] > T_elem['tra'][i]: titan.axvline(t[i])

plt.show()