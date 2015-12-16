from scipy.integrate import odeint
from scipy.linalg import svd, hankel
from scipy.signal import argrelmin, argrelmax
import numpy as np
from numpy import dot, cross, pi, isclose
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
import h5py
from tables.nodes import filenode
import dask.array as da
import dask.dataframe as dd
from dask.dataframe.rolling import wrap_rolling
import hashlib
from IPython.core.debugger import Tracer
from sympy.ntheory.factor_ import divisors

# Lengths in AU-converted-to-km, times in days, masses scaled to saturn (S_m)
#
# Most quantities are [Body]_[property], S being Saturn, T being Titan,
# H being Hyperion.
#
# NOTE:
# In general, an underscore at the end of a name means the norm of a vector
# quantity.

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
    """
    Transform from Euler angles to Wisdom angles
    """
    assert len(euler) == 3
    e_theta, e_phi, e_psi = euler[0:3]
    w_theta = atan((cos(e_theta)*sin(e_psi) + sin(e_theta)*cos(e_phi)*cos(e_psi))/\
        (cos(e_theta)*cos(e_phi)*cos(e_psi) - sin(e_theta)*sin(e_psi)))
    w_phi = atan(sin(e_phi)*cos(e_psi))
    w_psi = atan(-sin(e_phi)*sin(e_psi)/cos(e_phi))
    return [w_theta, w_phi, w_psi]

def dircos(wisdom, anom):
    """
    Calculate the directional cosines from a body towards Saturn, given its
    wisdom angles and true anomaly.
    """
    theta, phi, psi = wisdom[0:3]
    alpha = cos(theta - anom)*cos(psi) - sin(theta - anom)*sin(phi)*sin(psi)
    beta = sin(theta - anom)*cos(phi)
    gamma = cos(theta - anom)*sin(psi) + sin(theta - anom)*sin(phi)*cos(psi)
    return [alpha, beta, gamma]

def q_prod(q, r):
    """
    Find the Hamilton product of two quaternions
    """
    Q0 = q[0]*r[0] - dot(q[1:], r[1:])
    Q_1 = np.multiply(q[0], r[1:])
    Q_2 = np.multiply(r[0], q[1:])
    Q_ = np.add(np.add(Q_1, Q_2), cross(q[1:], r[1:]))
    return np.insert(Q_, 0, Q0)

def q_norm(q):
    """
    Find the norm of a quaternion
    """
    Q = [q[0], q[1]*1j, q[2]*1j, q[3]*1j]
    QC = [q[0], q[1]*-1j, q[2]*-1j, q[3]*-1j]
    return np.dot(Q, QC).real

def spc2bod(q, x):
    """
    Convert a quaternion from the space system to the body system
    """
    assert len(q) == len(x) == 4
    qc = [q[0], -q[1], -q[2], -q[3]]
    return q_prod(qc, q_prod(x, q))

def bod2spc(q, X):
    """
    Convert a quaternion from the body system to the space system
    """
    assert len(q) == len(X) == 4
    qc = [q[0], -q[1], -q[2], -q[3]]
    return q_prod(q, q_prod(X, qc))

def initialise():
    """
    Return a vector of initial values to use with the simulation
    """ 

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

    # Pure quaternion from the directional cosines
    H_q_0 = [0.0, 0.0, 0.0, 0.0]
    H_q_0[1:] = dircos(H_wisdom_0, H_anom_0)

    # Combine initial conditions into a handy vector
    return T_r_0 + H_r_0 + T_v_0 + H_v_0 + H_omega_0 + H_q_0

def rowdot(v1, v2):
    """
    Find the row-wise dot product of two arrays of 3-vectors
    """
    return np.einsum('ij, ij->i', v1, v2)

def dfdot(df1, df2):
    """
    Find the row-wise dot product of two DataFrames of 3-vectors
    """
    return pd.Series(np.einsum('ij, ij->i', df1, df2), index=df1.index)

def flattenacc(pos, R, J2):
    """
    Calculate the acceleration due to flattening given the J2-term
    """
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

def f(y, t0, titanic, flat):
    """
    Derivative Function: Return the necessary d/dt values when called by the
    integrator function in drive()
    """
    # First, split y into variables for the vectors it contains
    T_r, H_r, T_v, H_v, H_omega = \
    [y[i:i+3] for i in range(0, len(y)-4, 3)] 
    H_q = y[-4:]

    # Taking the norm of Hyperion's position vector and separation from Titan
    # now saves calculating it repeatedly later.
    H_r_ = norm(H_r)
    HT_sep = H_r - T_r
    HT_sep_ = norm(HT_sep)

    # Assign the flattening accelerations on Titan and Hyperion
    if flat:
        ST_flat = flattenacc(T_r, S_R, S_J2)
        SH_flat = flattenacc(H_r, S_R, S_J2)
        TH_flat = flattenacc(HT_sep, T_R, T_J2)
    else:
        ST_flat = SH_flat = TH_flat = [0.0, 0.0, 0.0]

    # Equations of translational motion for T and H
    T_a = -G * (S_m * T_r) / norm(T_r)**3 + ST_flat
    H_a = -G * ((S_m * H_r)/norm(H_r)**3 + \
          (T_m * HT_sep)/norm(HT_sep)**3) + \
          SH_flat + TH_flat

    # Read the directional cosines for Hyperion's principal axes w.r.t. Saturn
    # straight off the quaternion, and calculate them w.r.t. Titan
    H_dircos = H_q[1:] / np.sin(np.arccos(H_q[0]))
    HT_dircos = np.divide(HT_sep, HT_sep_)

    # Quaternion's EoM
    H_qdot = q_prod(bod2spc(H_q, np.insert(H_omega, 0, 0.0)), H_q) / 2

    # Include the influence of titan on the rotation of Hyperion. Or don't
    if titanic:
        HT_alpha = [(3*G*T_m/HT_sep_**3)*HT_dircos[1]*HT_dircos[2],
                    (3*G*T_m/HT_sep_**3)*HT_dircos[0]*HT_dircos[2],
                    (3*G*T_m/HT_sep_**3)*HT_dircos[0]*HT_dircos[1]]
    else:
        HT_alpha = [0.0, 0.0, 0.0]

    # Equation of motion for angular acceleration about H's principal axes.
    H_alpha = [H_BCA*(H_omega[1]*H_omega[2] - \
                   (3*G*S_m/H_r_**3)*H_dircos[1]*H_dircos[2] - HT_alpha[0]),
               H_CAB*(H_omega[0]*H_omega[2] - \
                   (3*G*S_m/H_r_**3)*H_dircos[0]*H_dircos[2] - HT_alpha[1]),
               H_ABC*(H_omega[0]*H_omega[1] - \
                   (3*G*S_m/H_r_**3)*H_dircos[0]*H_dircos[1] - HT_alpha[2])]

    return np.concatenate((T_v, H_v, T_a, H_a, H_alpha, H_qdot))


def drive(t_f=160, dt=0.001, chunksize=10000, titanic=True, flat=True, nrw=False, keep=50, path='output.h5'):
    """
    Run the actual integration!
    Just calling drive() takes the sim for a quick (160-day) spin
    t_f: Total simulation time (days)
    dt: Simulation timestep (days)
    chunksize: Size of chunks (timesteps, NOT NECESSARILY DAYS)
    titanic: Whether sim includes the influence of Ttian on the chaotic
        rotation of Hyperion (bool)
    path: Filepath to save output HDF5 file to (str)
    """

    if t_f/dt % chunksize != 0:
        print("Total number of timesteps must divide evenly into chunks.")
        return
    if t_f/dt % keep != 0 or chunksize % keep != 0:
        print("Timesteps and chunksize should both be divisible by keep")

    print("Running simulation to {} days in chunks of {:.0f} days."\
        .format(t_f, chunksize*dt), flush=1)
    print("Including" if titanic else "Ignoring",
        "the influence of Titan on the chaotic rotation of Hyperion.", flush=1)
    start = perf_counter() # Used to track time taken by entire sim
    y0 = initialise()
    # Create the column headings for the output DataFrame. quants is is a list
    # of 1st-level column labels, comps a list of subcolumn labels.
    quants = np.append(np.repeat(('T_r', 'H_r','T_v', 'H_v', 'H_omega'), 3),
                       np.repeat(('H_q'), 4))
    comps = np.append(np.tile(('x', 'y', 'z'), 4), # Cartesian elements
                      ('1', '2', '3', # H_omega
                       '0', '1', '2', '3')) # H_q

    ## Alternative columns for when we're just saving the angular bits
    #nrw_quants = np.append(np.repeat(('H_omega'), 3), np.repeat(('H_q'), 4))
    #nrw_comps = np.array(('1', '2', '3', '0', '1', '2', '3'))

    #t = np.arange(0, t_f, dt)
    
    # Start the DataFrame off with the initialisation vector so we have
    # something to save.
    df0 = pd.DataFrame(columns=[quants, comps], index=[0.0], dtype=np.float64)
    df0.iloc[0] = y0
    if nrw: df0 = df0[['H_omega', 'H_q']]
    with pd.HDFStore(path) as store:
        store.put('sim', df0, format='t', append=False) # t makes it appendable
        # trange() is a drop-in replacement for range() that renders a lovely
        # progress bar as we work through the range. 
        for i in trange(0, int(t_f / dt), chunksize, unit='chunk', leave=1):
            # The first time in the range given to odeint() must be the time on
            # y0. Because of this, each chunk's first row is the last element
            # of the previous chunk. Of course we then need an exception for
            # the first run, hence the ternary conditional in the t[] argument
            r, info = odeint(
                f, y0,
                np.linspace(0 if i==0 else (i-1)*dt, (i+chunksize)*dt, chunksize+(1 if i>0 else 0)),#t[0 if i==0 else i-1:i+chunksize],
                (titanic, flat),
                full_output=1
                )
            y0 = r[-1]
            jac = np.count_nonzero(info['mused']-1)
            if jac: print('\n', jac, flush=1)
            # We don't want to save the overlapping terms, though, so the
            # beginning of the slice on t used here is shifted by one element
            # from the one used for the integration.
            df = pd.DataFrame(
                r[::keep][1:],
                index=np.arange(i*dt,(i+chunksize)*dt,keep*dt)[1 if i==0 else 0:],#index=t[i+1 if i==0 else i:i+chunksize], 
                columns=[quants, comps],
                dtype=np.float64
                )
            if nrw: df = df[['H_omega', 'H_q']]
            store.append('sim', df)
            store.flush()

    end = perf_counter()
    print("\nSimulation successfully completed in {:.2f}s.\a".format(end-start))

def sep(store, a='H', b='T', key=None):
    """
    Calculate and save series of separation vectors between two bodies labelled
    by a and b
    """
    # Use body label strings to automatically name the table we store this in.
    if key is None: key = 'analysis/%s%ssep' % (a,b)
    # We here use string interpolation with the body labels to determine which
    # column in the sim table we're pulling data from. My gut says using string 
    # interp'n for this purpose is deeply wrong, but my head can't think of
    # a better way to do it. Expect to see this several more times
    df_sep = store.sim['%s_r' % a] - store.sim['%s_r' % b]
    store.put(key, df_sep)
    return store.select(key)

def ecc(store, body, key=None):
    """
    Calculate and save series of eccentricity vectors for the given body.
    Returns a reference to the series in the HDFStore.
    """
    # Use the body label str to automatically name the table we store this in.
    if key is None: key = 'analysis/%secc' % body
    # see comment in sep() for what the percents are doing here
    pos = store.sim['%s_r' % body]
    # This is the highest-performance way I've found to produce the norm of
    # every row in a dataframe.
    pos_ = np.sqrt(np.square(pos).sum(axis=1))
    vel = store.sim['%s_v' % body]
    # Now we're using string interpolation to assign variables! May God have
    # mercy on my soul.
    m = eval('%s_m' % body)
    df_ecc = cross(vel, cross(pos, vel))/(G*(S_m+m)) - pos.div(pos_, axis=0)
    store.put(key, df_ecc)
    return store.select(key)

def semi_major(store, body, key=None):
    """
    Calculate and save series of instantaneously-calculated semi-major axis
    values for the given body
    Returns a reference to the series in the HDFStore.
    """
    if key is None: key = 'analysis/%ssma' % body
    # see comment in sep() for what the percents are doing here
    pos = store.sim['%s_r' % body]
    pos = np.sqrt(np.square(pos).sum(axis=1))
    vel = store.sim['%s_v' % body]
    vel = np.sqrt(np.square(vel).sum(axis=1))
    mu = G * (S_m + eval('%s_m'%body))
    df_sma = 1/(2/pos - vel**2/mu)
    store.put(key, df_sma)
    return store.select(key)

def anom(store, body, e, key=None):
    """
    Calculate and save series of true anomalies for given body and
    instantaneous eccentricity vectors e.
    Returns a reference to the series in the HDFStore.
    """
    if key is None: key = 'analysis/%stra' % body
    # see comment in sep() for what the percents are doing here
    pos = store.sim['%s_r'%body]
    pos_ = np.sqrt(np.square(pos).sum(axis=1))
    vel = store.sim['%s_v'%body]
    e_ = np.sqrt(np.square(e).sum(axis=1))
    df_tra = np.arccos(dfdot(e, pos)/(e_*pos_))
    # Correct the signs on the true anomalies that need it. Query the DF for
    # relevant rows, then replace each one with 2pi minus itself.
    df_tra[dfdot(pos, vel) < 0] = df_tra[dfdot(pos, vel) < 0]\
        .apply(lambda x:2*pi-x)
    store.put(key, df_tra)
    return store.select(key)

def meanmot(store, body, sma, key=None):
    """
    Calculate and save series of instantaneously-calculated mean motions for
    given body and semi-major axis values sma.
    Returns a reference to the series in the HDFStore.
    """
    if key is None: key = 'analysis/%stra'%body
    mu = G * (S_m + eval('%s_m'%body))
    df_mm = np.sqrt(mu/sma**3)/(2*pi)
    store.put(key, df_mm)
    return store.select(key)

def orbits(path='output.h5', q=True, actual_seps=False):
    """
    Display readout showing orbital paths, basic stats, H-T separations over
    time and values of quaternion elements
    actual_seps: Whether we plot the actual instantaneous values of HT_sep, or
        just the rolling average and range-band. (bool)
    """
    with pd.HDFStore(path) as store:

        HT_sep = sep(store)
        HT_sep_ = np.sqrt(np.square(HT_sep).sum(axis=1)).values
        HT_sep_index = np.array(HT_sep.index, dtype=int)
        H_e = ecc(store, 'H')
        T_e = ecc(store, 'T')
        H_a = semi_major(store, 'H')
        T_a = semi_major(store, 'T')
        H_n = meanmot(store, 'H', H_a)
        T_n = meanmot(store, 'T', T_a)

        fig = plt.figure(figsize=(8, 10 if q else 8))
        fig.set_tight_layout(True)
        grid = gs.GridSpec(4 if q else 3, 3)
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
        # Take the running mean of separations using convolution, then NaN out
        # the values at either end so we don't get weird-looking lines.
        N = 1280
        sep_mean = np.convolve(HT_sep_, np.ones((N,))/N, mode='same')
        sep_mean[:N//2] = np.nan
        sep_mean[-N//2:] = np.nan
        # All this is a complicated way of making a band on the plot that 
        # should span the (potentially shifting) range of values of HT_sep.
        # Basically we find the value of the most extreme local extrema within
        # +/-32 days of each point (which should cover each 3:4 resonant cycle)
        # and plot the band between those max and min values at each timestep.
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
        # Plot the actual instantaneous values of HT_sep if we like
        if actual_seps:
            seps.plot(HT_sep_index, HT_sep_, alpha=0.5, color='#4C72B0')

        if q:
            quaternions = plt.subplot(grid[3, 0:3])
            quaternions.set_title('Elements of rotation quaternion of Hyperion')
            H_q = store.sim['H_q']
            for i in range(4):
                quaternions.plot(H_q.index, H_q[str(i)])
            quaternions.legend(('q₀', 'q₁', 'q₂', 'q₃'))

        info = plt.subplot(grid[0:2, -1])
        info.set_title('Info')
        info.axis('off')
        # This bit's a bit of a mess, but that's because the matplotlib Table
        # class is poorly written and even more poorly documented. We need a
        # list of label strings and then a list of value strings. Can't find a
        # good way of showing them lined up together in the code while still
        # being able to format the values, so you'll just have to pretend.
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
        # Whoever wrote this class seems to hate legibility, on top of their
        # many other professional failings. Thankfully, someone magnificent
        # human specimen on StackOverflow found out how to raise row heights:
        for c in tab.properties()['child_artists']:
            c.set_height(c.get_height()*2)

    plt.show()

def signchange(array):
    '''
    Takes an array of shape (2,), and returns whether or not there was a sign
    change between them.
    '''
    assert len(array) == 2
    a = array[0]
    b = array[1]
    if a < 0 and b >= 0:
        return True
    elif a >= 0 and b < 0:
        return True
    else:
        return False

def sect(var1, elem1, var2, elem2, var3, elem3, val=0, step=1,
         leaf1='sim', leaf2='sim', leaf3='sim', threed=1,
         path='output.h5'):
    with pd.HDFStore(path) as store:
        nrows = store[leaf1].shape[0]
        #xcols = pd.MultiIndex.from_tuples([(var1, elem1)])
        #ycols = pd.MultiIndex.from_tuples([(var2, elem2)])
        #zcols = pd.MultiIndex.from_tuples([(var3, elem3)])
        #x = dd.from_pandas(store[leaf1][var1][elem1], nrows//100)
        #y = dd.from_pandas(store[leaf1][var2][elem2], nrows//100)
        #z = dd.from_pandas(store[leaf1][var3][elem3], nrows//100)        
        x = store[leaf1][var1][elem1]
        y = store[leaf2][var2][elem2]
        if leaf3 == 'sim': 
            z = store[leaf3][var3][elem3]
        else:
            z = store[leaf3]
        #rolling_signchange = partial(pd.rolling_apply, func=signchange)
        #dask_signcheck = wrap_rolling(rolling_signchange)
        #idx = dask_signcheck(z, 2)
        #Tracer()()

        idx = np.sign((z-val).shift()) != np.sign(z - val)
        #idx.index = z.index
        #idx = dd.from_pandas(idx, parts)
        print('Plotting!', flush=1)
        fig = plt.figure(figsize=(8,8))
        fig.set_tight_layout(True)
        if not threed:
            ax = fig.add_subplot(111)
            ax.scatter(x[idx][::step], y[idx][::step], marker='+')
        else:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x[::step], y[::step], z[::step], marker='+')

        plt.show()
 


def h_svd(elem, n, J=None, m=3, theta=0, drop=1500, step=1, force=0, quant='omega', path='output.h5'):
    """
    Run Singular Value Decomposition on the quaternions. Currently eats all my
    RAM. All of it. These aren't the old days where you'd deal with three
    swap file purges before breakfast. If Mac OS X runs out of application
    memory it does not fail gracefully, but instead stumbles around like a 
    stunned cow until you put it out of its misery.
    i: (0,1,2,3) The element of the quaternion we're looking at
    n: Sample time (row length of trajectory matrix) (int, timesteps)
    J: Lag time (how many steps ahead of the preceding row each row of the
        trajectory matrix is) (int, timesteps)
        Default is J = n, as suggested by Richard and Tom's book. J=1 is the
        one suggested by Broomhead & King, but this seems to result in too
        much correlation between successive rows.
    drop: How many values at the beginning of the reconstructed timeseries to
        disregard (int)
    
    """

    if J is None: J = n

    with pd.HDFStore(path) as store:
         with h5py.File(path, 'r+') as h5pystore:
            data = store.sim.H_omega[str(elem)]
            quat = data.as_matrix()
            assert len(quat) % n == 0
            rows = (len(quat) - (n-J)) // J
            params = store.sim.tail(1).to_string() + \
                ''.join(map(str,[elem, n, J, m]))
            digest = hashlib.md5(params.encode()).hexdigest()
            if force or ('/svd/traj' not in h5pystore or \
                h5pystore['/svd/traj'].attrs["digest"] != np.string_(digest)):
                    print("Constructing new trajectory matrix...", flush=1)
                    if '/svd/traj' in h5pystore: del h5pystore['/svd']
                    traj = h5pystore.create_dataset('/svd/traj', (rows,n))
                    if J != n:
                        progress = tqdm(total=rows, leave=1)
                        for i, j in zip(range(rows), range(0, len(quat), J)):
                            traj[i] = quat[j:j+n]
                            progress.update()
                        progress.close()
                    else:
                        traj[:] = quat.reshape((rows, n))
                    traj.attrs["digest"] = np.string_(digest)
                    print()
            else:
                print("Reusing previous trajectory matrix...", flush=1)
                traj = h5pystore.require_dataset(
                    '/svd/traj', (rows, n), np.float32
                    )
            print("Constructing OoC trajectory and covariance matrices...", flush=1)
            datraj = da.from_array(traj, chunks=(1000, n))
            cov = da.dot(datraj.transpose(), datraj)
            #normn = len(quat) - (n-1)
            #traj = normn**-1/2 * hankel(quat, np.zeros(n))[::J]
            print("Running SVD...", flush=1)
            U, s, V = da.linalg.svd(cov)
            S = np.diag(s[0:m])
            #return(S[0,0], S[1,1], S[2,2])
            #print(traj.shape, cov.shape, U.shape, s.shape, V.shape)
            print(S)
            #recon = dot(U[:,0:m], dot(S, V[0:m,:]))
            #print("Writing U", flush=1)
            #h5pystore.create_dataset('/svd/U', data=U)
            ##print("Writing s", flush=1)
            ##h5pystore.create_dataset('/svd/s', data = S)
            #print("Writing V", flush=1)
            #h5pystore.create_dataset('/svd/V', data=V)
            #print("Writing cov", flush=1)
            #h5pystore.create_dataset('/svd/cov', data=cov)

            print("Taking Poincaré sections and projecting dataset...", flush=1)
            #plane_normal = np.cross(U[:,1], U[:,2])
            #plane_normal /= np.norm(plane_normal)
            x = dot(traj, U[:,0])
            y = dot(traj, U[:,1])
            z = dot(traj, U[:,2])
            idx = np.sign(np.roll(z, 1)) != np.sign(z)
            #idx = np.roll(z, 1) > z
            #idx = da.isclose(dist, 0)

            print("Plotting!", flush=1)
            fig = plt.figure(figsize=(8,8))
            fig.set_tight_layout(True)
            ax = fig.add_subplot(111)
            ax.scatter(traj[:,0][idx][::step], np.roll(traj[:,0][idx][::step], 1), marker='+')
            #ax.set_title("Poincare section of omega{} through plane of first two singular vectors".format(str(elem)))

            #embed = []
            #for i in range(m):
            #    embed.append(np.zeros(traj.shape[0]-drop))
            #    for j in range(traj.shape[0]-drop):
            #        embed[i][j] = np.inner(V[:,i], traj[j])

            #figatt = plt.figure(figsize=(8,8))
            #figatt.set_tight_layout(True)

            #axesatt = figatt.add_subplot(111)
            #axesatt.plot(embed[0], embed[1])

            plt.show()

def singspec(elem, path='output.h5'):
    with pd.HDFStore(path) as store:
        N = len(store.sim)
        divs = divisors(N)[4:12]

    vals = np.zeros((len(divs), 3))
    for i in range(len(divs)):
        print(divs[i], flush=1)
        vals[i] = h_svd(elem, divs[i], path=path)

    fig = plt.figure(figsize=(8,8))
    fig.set_tight_layout(True)
    grid = gs.GridSpec(3, 1)

    first = plt.subplot(grid[0])
    first.semilogx(divs, vals[:,0])

    second = plt.subplot(grid[1])
    second.semilogx(divs, vals[:,1])

    third = plt.subplot(grid[2])
    third.semilogx(divs, vals[:,2])

    plt.show()



def h5check(path='output.h5'):
    with pd.HDFStore(path) as store:
        print(store)


if __name__ == "__main__":
    drive()
    orbits()