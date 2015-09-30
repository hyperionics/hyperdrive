from scipy.integrate import ode
from numpy.linalg import norm
from numpy import subtract as sub
from numpy import add
import csv

# Lengths in AU, times in days, masses scaled to Mass_saturn (S_M)
#
# Most quantities are [Body]_[property], S being Saturn, T being Titan,
# H being Hyperion.

# 6.67408E-11 from m^3/(kg*s^2) to AU^3/(Mass of Saturn * day^2) gives:
G = 8.46E-8

# Masses
S_M = 1
T_M = 2.367E-4
H_M = 9.8E-9

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

def f(t, y):
    """Vector of Titan's velocity, Hyperion's velocity, T's acc, H's acc"""
    T_r = y[0:3]
    T_v = y[6:9]
    H_r = y[3:6]
    H_v = y[9:12]
    TH_sep = sub(T_r, H_r)
    T_a = [i * (G * S_M) / (T_M * norm(T_r)**3) for i in T_r]
    H_a = add([i * (G * S_M) / (H_M * norm(H_r)**3) for i in H_r],
        [i * (G * T_M) / (H_M * norm(TH_sep)**3) for i in TH_sep])
    vec = T_v + H_v + T_a + H_a
    return vec

# Initial and final times and timestep
t_i = 0
t_f = 100
dt = 0.2

drive = ode(f).set_integrator('dopri5')
drive.set_initial_value(T_r_0 + H_r_0 + T_v_0 + H_v_0, t_i)

with open('output.csv', 'w', newline='') as file:
    out = csv.writer(file)
    out.writerow(['t', 'Tx', 'Ty', 'Tz', 'TVx', 'TVy', 'TVz', 'Hx', 'Hy', 'Hz',
                 'HVx', 'HVy', 'HVz', 'THsep'])
    while drive.successful() and drive.t < t_f:
        result = drive.integrate(drive.t + dt)
        print(drive.t, drive.y)
        out.writerow(drive.t, drive.y)
