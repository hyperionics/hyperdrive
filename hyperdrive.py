from scipy.integrate import odeint, ode
from scipy.linalg import svd, hankel
from scipy.signal import argrelmin, argrelmax
import numpy as np
from numpy import dot, cross, pi
from numpy.linalg import norm
import pandas as pd
from pandas.tools.plotting import lag_plot
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gs
from mpl_toolkits.mplot3d import Axes3D
from math import sqrt, sin, cos, tan, asin, acos, atan, atan2, copysign
from functools import partial
from itertools import repeat, chain
from tqdm import trange, tqdm
from time import perf_counter
import seaborn as sns
from IPython.core.debugger import Tracer


# Lengths in AU-converted-to-km, times in days, masses scaled to saturn (S_m)
#
# Most quantities are [Body]_[property], S being Saturn, T being Titan,
# H being Hyperion.

AU = 149597871

# 6.67408E-11 from m^3/(kg*s^2) to AU^3/(Mass of Saturn * day^2) gives:
G = 8.46E-8 * AU**3

# Are we including the effect of Titan on the chaotic rotation of Hyperion?
titanic = True

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

# Dimensionless MoI
H_A = 0.314
H_B = 0.474
H_C = 0.542
H_BCA = (H_B - H_C)/H_A
H_CAB = (H_C - H_A)/H_B
H_ABC = (H_A - H_B)/H_C

T_A = T_B = T_C = 0.3414

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
    w_theta, w_phi, w_psi = wisdom[0:3]
    e_theta = atan((cos(w_theta)*sin(w_psi) + sin(w_theta)*sin(w_phi)*cos(w_psi))/\
        (cos(w_theta)*sin(w_phi)*cos(w_psi) - sin(w_theta)*sin(w_psi)))
    e_phi = atan(1/(cos(w_phi)*cos(w_psi)))
    e_psi = atan(-cos(w_phi)*sin(w_psi)/sin(w_phi))
    return [e_theta, e_phi, e_psi]

def dircos(wisdom, anom):
    """Calculate the directional cosines from a body towards Saturn"""
    theta, phi, psi = wisdom[0:3]
    alpha = cos(theta - anom)*cos(psi) - sin(theta - anom)*sin(phi)*sin(psi)
    beta = sin(theta - anom)*cos(phi)
    gamma = cos(theta - anom)*sin(psi) + sin(theta - anom)*sin(phi)*cos(psi)
    return [alpha, beta, gamma]

def q_prod(q, r):
    """Calculate Hamilton product of two quaternions"""
    Q0 = q[0]*r[0] - dot(q[1:], r[1:])
    Q_1 = [q[0] * i for i in r[1:]]
    Q_2 = [r[0] * i for i in q[1:]]
    Q_ = np.add(np.add(Q_1, Q_2), cross(q[1:], r[1:]))
    return np.concatenate(([Q0], Q_))

def q_norm(q):
    """Calculate the norm of two quaternons"""
    Q = [q[0], q[1]*1j, q[2]*1j, q[3]*1j]
    QC = [q[0], q[1]*-1j, q[2]*-1j, q[3]*-1j]
    return np.dot(Q, QC).real

def spc2bod(q, x):
    """Switch from space- to body-system quaternions"""
    assert len(q) == len(x) == 4
    qc = [q[0], -q[1], -q[2], -q[3]]
    return q_prod(qc, q_prod(x, q))

def bod2spc(q, X):
    """Switch from body- to space-system quaternions"""
    assert len(q) == len(X) == 4
    qc = [q[0], -q[1], -q[2], -q[3]]
    return q_prod(q, q_prod(X, qc))

def initialise():
    #Initial values for the positions and velocities, from HORIZONS on 2005-09-25
    T_r_0 = [-4.088407843090480E-03 *AU,
             -6.135746299499617E-03 *AU,
             3.570710328903993E-03 *AU]

    H_r_0 = [-9.556008109223760E-03 *AU,
             -6.330869216667536E-04 *AU,
             1.293277241526503E-03 *AU]

    T_v_0 = [2.811008677848850E-03 *AU,
             -1.470816650319267E-03 *AU,
             4.806729268010478E-04 *AU]

    H_v_0 = [6.768811686476851E-04 *AU,
             -2.639837861571281E-03 *AU,
             1.253587766343593E-03 *AU]

    # Initials used by Sinclair et al.
    # T_r_0 = [-0.0075533871, 0.0025250254, -0.0000462204]
    # T_v_0 = [-0.0010017342, -0.0031443009, 0.0000059503]
    # H_r_0 = [-0.0006436995, 0.0099145485, 0.0000357506]
    # H_v_0 = [-0.0029182723, 0.0000521415, -0.0000356145]

    # From Harbison, Cassini flyby on 2005-09-25
    H_omega_0_mag = 72*pi/180
    H_omega_0 = [i * H_omega_0_mag for i in [0.902, 0.133, 0.411]]

    # From Harbison, Cassini flyby on 2005-09-25
    H_euler_0 = [2.989, 1.685, 1.641]
    H_wisdom_0 = wis(H_euler_0)
    H_anom_0 = 2.780523844083934E+02

    H_q_0 = [0.0, 0.0, 0.0, 0.0]
    H_q_0[1:] = dircos(H_wisdom_0, H_anom_0)

    # Combine initial conditions into a handy vector
    return T_r_0 + H_r_0 + T_v_0 + H_v_0 + \
        H_euler_0 + H_omega_0 + H_omega_0 + H_q_0

wise = 0

def rowdot(v1, v2):
    """Row-wise dot product of two arrays"""
    return np.einsum('ij, ij->i', v1, v2)

def dfdot(df1, df2):
    """Row-wise dot product of two dataframes"""
    return pd.Series(np.einsum('ij, ij->i', df1, df2), index=df1.index)

def wisdom_derivs(wisdom, omega):
    """Calculate the time-derviatives of the euler angles due to ang.vel."""
    theta, phi, psi = wisdom[0:3]
    Dtheta = (omega[2]*cos(psi) - omega[0]*sin(psi))/cos(phi)
    Dphi = omega[2]*sin(psi) + omega[0]*cos(psi)
    Dpsi = omega[1] - Dtheta*sin(phi)
    return [Dtheta, Dphi, Dpsi]

def euler_derivs(euler, omega):
    """Calculate the time-derivatives of the wisdom angles due to ang. vel."""
    theta, phi, psi = euler[0:3]
    Dtheta = (omega[0]*sin(psi) + omega[1]*cos(psi))/sin(phi)
    Dphi = omega[0]*cos(psi) - omega[1]*sin(psi)
    Dpsi = omega[2] - Dtheta*cos(phi)
    return [Dtheta, Dphi, Dpsi]

def euler_dircos(euler, nu):
    """Directional cosines based on euler angles"""
    theta, phi, psi = euler[0:3]
    alpha = cos(theta - nu)*cos(psi) - sin(theta - nu)*cos(phi)*sin(psi)
    beta = cos(theta - nu)*sin(-psi) - sin(theta - nu)*cos(phi)*cos(psi)
    gamma = sin(theta - nu)*sin(phi)
    return [alpha, beta, gamma]

def wisdom_dircos(wisdom, nu):    
    """Directional cosines based on wisdom angles"""
    theta, phi, psi = wisdom[0:3]
    alpha = cos(theta - nu)*cos(psi) - sin(theta - nu)*sin(phi)*sin(psi)
    beta = sin(theta - nu)*cos(phi)
    gamma = cos(theta - nu)*sin(psi) + sin(theta - nu)*sin(phi)*cos(psi)
    return [alpha, beta, gamma]

def flattenacc(pos, R, J2):
    """Acceleration due to flattening terms"""
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

def f(t0, y, titanic):
    """Vector of Titan's velocity, Hyperion's velocity, T's acc, H's acc"""
    global wise

    [T_r, H_r, T_v, H_v, H_angle, H_a_omega, H_q_omega] = \
    [y[i:i+3] for i in range(0, len(y)-4, 3)]
    H_q = y[-4:]

    H_r_ = norm(H_r)
    HT_sep = H_r - T_r
    HT_sep_ = norm(HT_sep)

    ST_flat = flattenacc(T_r, S_R, S_J2)
    assert acos(dot(ST_flat, -T_r)/(norm(ST_flat)*norm(T_r))) <= pi/2
    SH_flat = flattenacc(H_r, S_R, S_J2)
    assert acos(dot(SH_flat, -H_r)/(norm(SH_flat)*norm(H_r))) <= pi/2
    TH_flat = flattenacc(HT_sep, T_R, T_J2)
    assert acos(dot(TH_flat, -HT_sep)/(norm(TH_flat)*norm(HT_sep))) <= pi/2

    T_a = -G * (S_m * T_r) / norm(T_r)**3 + ST_flat
    H_a = -G * ((S_m * H_r)/norm(H_r)**3 + \
          (T_m * HT_sep)/norm(HT_sep)**3) + \
          SH_flat + TH_flat

    ecc = cross(H_v, cross(H_r, H_v))/(G*S_m) - H_r/norm(H_r)
    anom = acos(dot(ecc, H_r)/(norm(ecc)*norm(H_r)))
    if dot(H_r, H_v) < 0: anom = 2*pi - anom

    H_angle = np.arctan2(np.sin(H_angle), np.cos(H_angle))

    if not wise:
        if abs(sin(H_angle[1])) <= 0.1:
            print('Switching to Wisdom!', flush=True)
            H_angle = wis(H_angle)
            wise = True
            H_a_dircos = wisdom_dircos(H_angle, anom)
            H_adot = wisdom_derivs(H_angle, H_a_omega)
        else:
            H_a_dircos = euler_dircos(H_angle, anom)
            H_adot = euler_derivs(H_angle, H_a_omega)
    else:
        if abs(cos(H_angle[1])) <= 0.1:
            print('Switching to Euler!', flush=True)
            H_angle = eul(H_angle)
            wise = False
            H_a_dircos = euler_dircos(H_angle, anom)
            H_adot = euler_derivs(H_angle, H_a_omega)
        else:
            H_a_dircos = wisdom_dircos(H_angle, anom)
            H_adot = wisdom_derivs(H_angle, H_a_omega)

    H_q_dircos = H_q[1:]
    HT_dircos = [HT_sep[i]/HT_sep_ for i in range(0,3)]

    H_qdot = [(1/2)*i for i in q_prod(bod2spc(H_q,
        np.concatenate(([0.0], H_q_omega))), H_q)]

    # Include the influence of titan on the rotation of Hyperion. Or don't.
    if titanic:
        HT_alpha = [(3*G*T_m/HT_sep_**3)*HT_dircos[1]*HT_dircos[2],
                    (3*G*T_m/HT_sep_**3)*HT_dircos[0]*HT_dircos[2],
                    (3*G*T_m/HT_sep_**3)*HT_dircos[0]*HT_dircos[1]]
    else:
        HT_alpha = [0.0, 0.0, 0.0]

    H_q_alpha = [H_BCA*(H_q_omega[1]*H_q_omega[2] - \
                   (3*G*S_m/H_r_**3)*H_q_dircos[1]*H_q_dircos[2] - HT_alpha[0]),
               H_CAB*(H_q_omega[0]*H_q_omega[2] - \
                   (3*G*S_m/H_r_**3)*H_q_dircos[0]*H_q_dircos[2] - HT_alpha[1]),
               H_ABC*(H_q_omega[0]*H_q_omega[1] - \
                   (3*G*S_m/H_r_**3)*H_q_dircos[0]*H_q_dircos[1] - HT_alpha[2])]

    H_a_alpha = [H_BCA*(H_a_omega[1]*H_a_omega[2] - \
                   (3*G*S_m/H_r_**3)*H_a_dircos[1]*H_a_dircos[2] - HT_alpha[0]),
               H_CAB*(H_a_omega[0]*H_a_omega[2] - \
                   (3*G*S_m/H_r_**3)*H_a_dircos[0]*H_a_dircos[2] - HT_alpha[1]),
               H_ABC*(H_a_omega[0]*H_a_omega[1] - \
                   (3*G*S_m/H_r_**3)*H_a_dircos[0]*H_a_dircos[1] - HT_alpha[2])]

    vec = np.concatenate((T_v, H_v, T_a, H_a, H_adot, H_a_alpha, H_q_alpha, H_qdot))
    return vec

def drive(t_f=160, dt=0.001, chunksize=10000, titanic=True, path='output.h5'):
    print("Running simulation to {} days in chunks of {:.0f} days."\
        .format(t_f, chunksize*dt), flush=1)
    print("Including" if titanic else "Ignoring",
        "the influence of Titan on the chaotic rotation of Hyperion.", flush=1)
    start = perf_counter()
    y0 = initialise()

    t = np.arange(0, t_f, dt)
    assert len(t) % chunksize == 0, \
        "Total number of timesteps must divide evenly into chunks"

    quants = np.append(np.repeat(('T_r', 'H_r', 'T_v', 'H_v',
                                  'H_angle', 'H_a_omega', 'H_q_omega'), 3),
                       np.repeat(('H_q'), 4))
    comps = np.concatenate((np.tile(('x', 'y', 'z'), 4), # Cartesian elements
                      np.tile(('1', '2', '3'), 3), # Angular elements
                      np.array(('0', '1', '2', '3')))) # H_q
    
    df0 = pd.DataFrame(columns=[quants, comps], index=[0.0], dtype=np.float64)
    df0.iloc[0] = y0

    integrator = ode(f)
    integrator.set_f_params((titanic,))
    integrator.set_integrator('dopri5')
    integrator.set_initial_value(y0)

    with pd.HDFStore(path) as store:
        store.put('sim', df0, format='t', append=False)
        progress = tqdm(total=len(t)//dt, leave=1)
        for i in range(0, len(t), chunksize):
            # r = odeint(f, y0,
            #            t[i if i==0 else i-1:i+chunksize],
            #            (titanic,))
            r = np.empty((0, len(quants)), float)
            for j in t[i if i==0 else i-1:i+chunksize]:
                r = np.vstack((r, integrator.integrate(j+dt)))
                if j % (chunksize//10): progress.update(chunksize//10)
            # y0 = r[-1]
            df = pd.DataFrame(
                r[1:],
                index=t[i+1 if i==0 else i:i+chunksize],
                columns=[quants, comps],
                dtype=np.float64
                )
            store.append('sim', df)
            store.flush()
        progress.close()

    end = perf_counter()
    print("\nSimulation successfully completed in {:.2f}s.".format(end-start))

def sep(store, a='H', b='T', key=None):
    if key is None: key = 'analysis/%s%ssep' % (a,b)
    df_sep = store.sim['%s_r' % a] - store.sim['%s_r' % b]
    store.put(key, df_sep)
    return store.select(key)

def ecc(store, body, key=None):
    if key is None: key = 'analysis/%secc' % body
    pos = store.sim['%s_r' % body]
    pos_ = np.sqrt(np.square(pos).sum(axis=1))
    vel = store.sim['%s_v' % body]
    m = eval('%s_m' % body)
    df_ecc = cross(vel, cross(pos, vel))/(G*(S_m+m)) - pos.div(pos_, axis=0)
    store.put(key, df_ecc)
    return store.select(key)

def semi_major(store, body, key=None):
    if key is None: key = 'analysis/%ssma' % body
    pos = store.sim['%s_r' % body]
    pos = np.sqrt(np.square(pos).sum(axis=1))
    vel = store.sim['%s_v' % body]
    vel = np.sqrt(np.square(vel).sum(axis=1))
    mu = G * (S_m + eval('%s_m'%body))
    df_sma = 1/(2/pos - vel**2/mu)
    store.put(key, df_sma)
    return store.select(key)

def anom(store, body, e, key=None):
    if key is None: key = 'analysis/%stra' % body
    pos = store.sim['%s_r'%body]
    pos_ = np.sqrt(np.square(pos).sum(axis=1))
    vel = store.sim['%s_v'%body]
    e_ = np.sqrt(np.square(e).sum(axis=1))
    df_tra = np.arccos(dfdot(e, pos)/(e_*pos_))
    # This ought to correct the signs
    df_tra[dfdot(pos, vel) < 0] = df_tra[dfdot(pos, vel) < 0]\
        .apply(lambda x:2*pi-x)
    store.put(key, df_tra)
    return store.select(key)

def meanmot(store, body, sma, key=None):
    if key is None: key = 'analysis/%stra'%body
    mu = G * (S_m + eval('%s_m'%body))
    df_mm = np.sqrt(mu/sma**3)/(2*pi)
    store.put(key, df_mm)
    return store.select(key)

def orbits(path='output.h5'):

    with pd.HDFStore(path) as store:

        HT_sep = sep(store)
        HT_sep_ = np.sqrt(np.square(HT_sep).sum(axis=1)).values[::1000]
        HT_sep_index = np.array(HT_sep.index[::1000], dtype=int)
        H_e = ecc(store, 'H')
        T_e = ecc(store, 'T')
        H_a = semi_major(store, 'H')
        T_a = semi_major(store, 'T')
        H_n = meanmot(store, 'H', H_a)
        T_n = meanmot(store, 'T', T_a)

        fig = plt.figure(figsize=(8, 10))
        fig.set_tight_layout(True)
        grid = gs.GridSpec(4, 3)
        plt.rcParams['axes.formatter.limits'] = [-5,5]

        orbits = plt.subplot(grid[0:2, 0:2])
        orbits.set_title('Path of simulated orbits')
        T_r = store.sim['T_r']
        H_r = store.sim['H_r']
        orbits.plot(T_r.x, T_r.y, H_r.x, H_r.y)
        orbits.plot(0,0, 'xr')
        orbits.axis([-2E6, 2E6, -2E6, 2E6])
        orbits.set_xlabel('km')
        orbits.set_ylabel('km')
        orbits.legend(('Titan', 'Hyperion'))

        seps = plt.subplot(grid[2, 0:3])
        seps.set_title('Magnitude of separation between Titan and Hyperion')
        seps.set_xlabel('days')
        seps.set_ylabel('km')
        N = 64
        sep_mean = np.convolve(HT_sep_, np.ones((N,))/N, mode='same')
        sep_mean[:N//2] = np.nan
        sep_mean[-N//2:] = np.nan
        sep_min = []
        sep_max = []
        sep_min_idx = argrelmin(HT_sep_, order=N//2, mode='clip')[0]
        sep_max_idx = argrelmax(HT_sep_, order=N//2, mode='clip')[0]
        for i in range(len(HT_sep_)):
            nearest_min_idx = sep_min_idx[(np.abs(sep_min_idx - i)).argmin()]
            sep_min.append(HT_sep_[nearest_min_idx])
            nearest_max_idx = sep_max_idx[(np.abs(sep_max_idx - i)).argmin()]
            sep_max.append(HT_sep_[nearest_max_idx])
        seps.plot(HT_sep_index, sep_mean)
        seps.fill_between(HT_sep_index, sep_min, sep_max,
                          alpha=0.3)
        # seps.plot(HT_sep_index, HT_sep_, alpha=0.5, color='#4C72B0')

        quaternions = plt.subplot(grid[3, 0:3])
        quaternions.set_title('Elements of rotation quaternion of Hyperion')
        H_q = store.sim['H_q']
        for i in range(4):
            quaternions.plot(H_q.index, H_q[str(i)])

        info = plt.subplot(grid[0:2, -1])
        info.set_title('Info')
        info.axis('off')
        labels = [
            'Hyp init e',
            'Hyp final e',
            'Tit init e',
            'Tit final e',
            'Init n/n',
            'Final n/n',
            'Max separation',
            'Min separation'
            ]
        text = [["{:.3g}".format(i)] for i in [
            norm(H_e.iloc[0]), # We round these to 3sf...
            norm(H_e.iloc[-1]),
            norm(T_e.iloc[0]),
            norm(T_e.iloc[-1]),
            H_n.iloc[0]/T_n.iloc[0],
            H_n.iloc[-1]/T_n.iloc[-1],
            HT_sep_.max(), # And show these in exponential form
            HT_sep_.min()
            ]
        ]
        tab = info.table(rowLabels=labels,
                   cellText=text,
                   loc='upper right',
                   colWidths=[0.5]*2)
        # Whoever wrote the Table class hates legibility. 
        # Let's increase the row height:
        for c in tab.properties()['child_artists']:
            c.set_height(c.get_height()*2)

    plt.show()

def poincare(path='output.h5'):
    with pd.HDFStore(path) as store:
        fig = plt.figure(figsize=(12, 4))
        fig.set_tight_layout(True)
        grid = gs.GridSpec(1, 3)

        e = ecc(store, 'H')
        nu = anom(store, 'H', e)

        idx = nu.shift() > nu

        o = store.sim['H_q_omega']
        q = store.sim['H_q']

        wo = store.sim['H_a_omega']
        wt = store.sim['H_wisdom']

        a = plt.subplot(grid[0])
        a.scatter(wo['1'][idx], wt['1'][idx])
        b = plt.subplot(grid[1])
        b.scatter(wo['2'][idx], wt['2'][idx])
        c = plt.subplot(grid[2])
        c.scatter(wo['3'][idx], wt['3'][idx])
    plt.show()


def angle_test(path='output.h5'):
    with pd.HDFStore(path) as store:
        fig = plt.figure(figsize=(12, 9))
        fig.set_tight_layout(True)
        grid = gs.GridSpec(3, 1)
        H_a_omega = store.sim['H_a_omega']
        H_q_omega = store.sim['H_q_omega']

        a = plt.subplot(grid[0])
        a.plot(H_a_omega.index, H_a_omega)

        q = plt.subplot(grid[1])
        q.plot(H_q_omega.index, H_q_omega)

        d = plt.subplot(grid[2])
        d.plot(H_a_omega.index, H_a_omega - H_q_omega)

    plt.show()

def q_svd(i, n, J=None, drop=1500, path='output.h5'):

    #WORK IN PROGRESS
    if J is None: J = n

    with pd.HDFStore(path) as store:
        quat = store.sim.H_q[str(i)]
        assert len(quat) % n == 0
        normn = len(quat) - (n-1)
        traj = normn**-1/2
        hankel(quat, np.zeros(n))[::J]
        U, s, V = svd(traj, False)
        m = 2 # embedding dimension
        S = np.diag(s[0:m])
        print(s)
        recon = dot(U[:,0:m], dot(S, V[0:m,:]))

        fig = plt.figure(figsize=(4,4))
        fig.set_tight_layout(True)

        axes = plt.plot(quat.index[::J], recon[:,0], quat.index[::J], quat[::J])

        embed = []
        for i in range(m):
            embed.append(np.zeros(traj.shape[0]-drop))
            for j in range(traj.shape[0]-drop):
                embed[i][j] = np.inner(V[:,i], traj[j])

        figatt = plt.figure(figsize=(8,8))
        figatt.set_tight_layout(True)

        axesatt = figatt.add_subplot(111)
        axesatt.plot(embed[0], embed[1])

        plt.show()

if __name__ == "__main__":
    drive()