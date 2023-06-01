from utils.util_functions import SignApprox
from utils.MotionPlanning import CreateDesiredTrajectory
import numpy as np

# Dynamic Params
global g, dt
global l, b, d
global Jx, Jy, Jz
global a1, a2, a3
global b1, b2, b3

g = 9.81
l = 0.23
Jx = 7.5e-3
Jy = 7.5e-3
Jz = 1.3e-2
Jr = 1e-5
b = 3.13e-5
d = 7.5e-5
a1 = (Jy - Jz) / Jx
a2 = (Jz - Jx) / Jy
a3 = (Jx - Jy) / Jz
b1 = l / Jx
b2 = l / Jy
b3 = 1 / Jz
dt = 0.01


# X_desired = np.append(X_desired,np.array([Phid,Ttad,Psid,Zd,Xd,Yd]))
# Parameter Definition
# global S, S_dot
# global c1, c2, c3, c4, c5, c6
# global k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12

def SMC(t, x, y, Uxy, m):
    # global X_desired

    # Actual States
    Phi = x[0]
    Phi_dot = x[1]
    Phi_ddot = y[0]
    Tta = x[2]
    Tta_dot = x[3]
    Tta_ddot = y[1]
    Psi = x[4]
    Psi_dot = x[5]
    Psi_ddot = y[2]

    Z = x[6]
    Z_dot = x[7]
    Z_ddot = y[3]
    X = x[8]
    X_dot = x[9]
    X_ddot = y[4]
    Y = x[10]
    Y_dot = x[11]
    Y_ddot = y[5]

    # Desired States
    DesPos, DesAtt = CreateDesiredTrajectory(t)
    # print(DesPos)

    Zd = DesPos[0]
    Zd_dot = DesPos[1]
    Zd_ddot = DesPos[2]
    Xd = DesPos[3]
    Xd_dot = DesPos[4]
    Xd_ddot = DesPos[5]
    Yd = DesPos[6]
    Yd_dot = DesPos[7]
    Yd_ddot = DesPos[8]

    Phid = DesAtt[0]
    Phid_dot = DesAtt[1]
    Phid_ddot = DesAtt[2]
    Ttad = DesAtt[3]
    Ttad_dot = DesAtt[4]
    Ttad_ddot = DesAtt[5]
    Psid = DesAtt[6]
    Psid_dot = DesAtt[7]
    Psid_ddot = DesAtt[8]

    # Altitude Control
    k7 = 5
    k8 = 3
    c4 = 5

    e4 = Zd - Z
    e4_dot = Zd_dot - Z_dot
    s4 = e4_dot + c4 * e4

    U1 = m / np.cos(Phi) / np.cos(Tta) * (k7 * SignApprox(s4, 3, 2) + k8 * s4 + g + Zd_ddot + c4 * e4_dot)
    # x Motion Control
    k9 = 5
    k10 = 1
    c5 = 5

    e5 = Xd - X
    e5_dot = Xd_dot - X_dot
    s5 = e5_dot + c5 * e5

    Ux = m / U1 * (k9 * SignApprox(s5, 3, 2) + k10 * s5 + Xd_ddot + c5 * e5_dot)
    # print(Ux)
    Ux_dot = (Ux[0] - Uxy[0]) / dt

    # y Motion Control
    k11 = 5
    k12 = 1
    c6 = 5

    e6 = Yd - Y
    e6_dot = Yd_dot - Y_dot
    s6 = e6_dot + c6 * e6

    Uy = m / U1 * (k11 * SignApprox(s6, 3, 2) + k12 * s6 + Yd_ddot + c6 * e6_dot)
    Uy_dot = (Uy[0] - Uxy[1]) / dt

    # Roll Control
    k1 = 3
    k2 = 0
    c1 = 10

    e1 = -Uy - Phi
    e1_dot = -Uy_dot - Phi_dot
    s1 = e1_dot + c1 * e1

    U2 = 1 / b1 * (k1 * SignApprox(s1, 3, 2) + k2 * s1 - a1 * Tta_dot * Psi_dot + Phid_ddot + c1 * e1_dot)

    # Pitch Control
    k3 = 3
    k4 = 0
    c2 = 10

    e2 = Ux - Tta
    e2_dot = Ux_dot - Tta_dot
    s2 = e2_dot + c2 * e2

    U3 = 1 / b2 * (k3 * SignApprox(s2, 3, 2) + k4 * s2 - a2 * Phi_dot * Psi_dot + Ttad_ddot + c2 * e2_dot)

    # Yaw Control
    k5 = 0.1
    k6 = 5
    c3 = 5

    e3 = Psid - Psi
    e3_dot = Psid_dot - Psi_dot
    s3 = e3_dot + c3 * e3

    U4 = 1 / b3 * (k5 * SignApprox(s2, 3, 2) + k6 * s3 - a3 * Phi_dot * Tta_dot + Psid_ddot + c3 * e3_dot)

    # Control Inputs
    U = np.array([U1, U2, U3, U4])
    Uxy = np.array([Ux[0], Uy[0]])

    # Save Slide Surfaces
    e3_ddot = Psid_ddot - Psi_ddot
    s3_dot = e3_ddot + c3 * e3_dot
    e4_ddot = Zd_ddot - Z_ddot
    s4_dot = e4_ddot + c4 * e4_dot
    e5_ddot = Xd_ddot - X_ddot
    s5_dot = e5_ddot + c5 * e5_dot
    e6_ddot = Yd_ddot - Y_ddot
    s6_dot = e6_ddot + c6 * e6_dot

    # S       = np.append(S,    np.array([s3,s4,s5,s6]))
    # S_dot   = np.append(S_dot,np.array([s3_dot,s4_dot,s5_dot,s6_dot]))

    return U, Uxy
