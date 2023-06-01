"""
March 12, 2022
Creator: Iman Sharifi
"""
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

self_tuning = 1
Uncertainty = 0
switching = 0
dist = 0

import sys
from math import isnan, pi
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

import torch
import torch.nn as nn
import torch.optim as optim

from utils.Quadcopter_Dynamics import Quadcopter_Dynamics_with_Disturbance, Add_Noise
from utils.A2C_models.A2C_model import NewAgent
from utils import QuadParams
from utils.util_functions import Saturation, Reward, reset_parameters, reset_params
from utils.SMC_Controller import SMC

Parameters = QuadParams


class my_struct:
    min = 0
    max = 0


def GainBound(static_gain, coef1=0.3, coef2=2):
    out = my_struct()
    out.min, out.max = static_gain * coef1, static_gain * coef2
    return out


global dt, T, Tf
# -------------------  Model Dynamics Parameters  ---------------------------------#
print("Model Parameters ...")
global m, g
global l, b, d
global Jx, Jy, Jz
global a1, a2, a3
global b1, b2, b3

m, g, l = Parameters.m, Parameters.g, Parameters.l
Jx, Jy, Jz = Parameters.Jx, Parameters.Jy, Parameters.Jz
b, d = Parameters.b, Parameters.d
a1, a2, a3 = Parameters.a1, Parameters.a2, Parameters.a3
b1, b2, b3 = Parameters.b1, Parameters.b2, Parameters.b3

print("Initialize Requirements ...")
dt = 0.01
T = dt
Tf = 100
t = np.arange(0, Tf + dt, dt)
N = int(Tf / dt) + 1
n = 12 + 3
x0 = np.zeros((n, 1))
x = x0
Xs = np.zeros((N + 1, n))
y = np.zeros((6, 1))
Uxy = np.array([0, 0])

U = np.zeros((1, 4))
u = np.zeros((4, 1))

Xd = np.zeros((N + 1, n))
Xm = np.zeros((N + 1, n))
m0 = m
M = m0 * np.ones((N + 1, 1))

# ------------Reinforcement Learning Parameters ---------------------------------
# Value functions------------ Rewards---------------Temporal differense Error----
V_z = np.zeros((N + 1, 1))
R_z = np.zeros((N + 1, 1))
delta_z = np.zeros((N + 1, 1))
V_phi = np.zeros((N + 1, 1))
R_phi = np.zeros((N + 1, 1))
delta_phi = np.zeros((N + 1, 1))
V_tta = np.zeros((N + 1, 1))
R_tta = np.zeros((N + 1, 1))
delta_tta = np.zeros((N + 1, 1))
V_psi = np.zeros((N + 1, 1))
R_psi = np.zeros((N + 1, 1))
delta_psi = np.zeros((N + 1, 1))

# --------- Create Desired Signal for tracking -----------------#
# ------Track a Constant point in 3D world
# Xd[:-1, 6] = 1
# Xd[:-1, 8] = 1
# Xd[:-1,10] = 2

# ------- Track a Square path in a constant altitude in 3D world
Xd[:-1, 4] = np.deg2rad(10 * signal.square(5 * np.pi / Tf * t))
Xd[:, 6] = 1
Xd[:-1, 8] = 1 + signal.square(4 * np.pi / Tf * t)
Xd[:-1, 10] = 1 + signal.square(4 * np.pi / Tf * (t - Tf / 8))

# ----Track a Helical path in 3D world
# Xd[:-1,  4] = np.deg2rad(np.sin(5*np.pi/Tf*t))
# Xd[:-1,  6] = 0.05*t
# Xd[:-1,  8] = np.cos(4*np.pi/Tf*t)  
# Xd[:-1, 10] = np.sin(4*np.pi/Tf*t)

# Subsystems bounds for z
Sigma_domain_z = 0.5
Sigma_domain_phi, Sigma_domain_tta, Sigma_domain_psi = pi / 36.0, pi / 36.0, pi / 36.0

# -----------  PID Control Initialization ------------------------#

# -----------------  Static Gains   -----------------------------------#

print("Initialize PID Params ...")
Kp_x_static = 7 * 3.1415 / 180
Ki_x_static = 0
Kd_x_static = 7 * 3.1415 / 180

Kp_y_static = 7 * 3.1415 / 180
Ki_y_static = 0
Kd_y_static = 7 * 3.1415 / 180

Kp_z_static = 15
Ki_z_static = 2
Kd_z_static = 15

Kp_psi_static = 3.0
Ki_psi_static = 0.0
Kd_psi_static = 0.3

Kp_phi_static = 0.5
Ki_phi_static = 0.0
Kd_phi_static = 0.2

Kp_tta_static = 0.5
Ki_tta_static = 0.0
Kd_tta_static = 0.2

# -------------------    Dynamic Gains  ------------------------------------#

Kp_x_dynamic = np.zeros((N + 1, 1))
Ki_x_dynamic = np.zeros((N + 1, 1))
Kd_x_dynamic = np.zeros((N + 1, 1))

Kp_y_dynamic = np.zeros((N + 1, 1))
Ki_y_dynamic = np.zeros((N + 1, 1))
Kd_y_dynamic = np.zeros((N + 1, 1))

Kp_z_dynamic = np.zeros((N + 1, 1))
Ki_z_dynamic = np.zeros((N + 1, 1))
Kd_z_dynamic = np.zeros((N + 1, 1))

Kp_psi_dynamic = np.zeros((N + 1, 1))
Ki_psi_dynamic = np.zeros((N + 1, 1))
Kd_psi_dynamic = np.zeros((N + 1, 1))

Kp_phi_dynamic = np.zeros((N + 1, 1))
Ki_phi_dynamic = np.zeros((N + 1, 1))
Kd_phi_dynamic = np.zeros((N + 1, 1))

Kp_tta_dynamic = np.zeros((N + 1, 1))
Ki_tta_dynamic = np.zeros((N + 1, 1))
Kd_tta_dynamic = np.zeros((N + 1, 1))

# ------------------  Main Gains  -------------------------------------#
Kp_x = np.zeros((N + 1, 1))
Kp_x_min = 0
Kp_x_max = 2 * Kp_x_static
Ki_x = np.zeros((N + 1, 1))
Ki_x_min = 0
Ki_x_max = 2 * Ki_x_static
Kd_x = np.zeros((N + 1, 1))
Kd_x_min = 0
Kd_x_max = 2 * Kd_x_static

Kp_y = np.zeros((N + 1, 1))
Kp_y_min = 0
Kp_y_max = 2 * Kp_y_static
Ki_y = np.zeros((N + 1, 1))
Ki_y_min = 0
Ki_y_max = 2 * Ki_y_static
Kd_y = np.zeros((N + 1, 1))
Kd_y_min = 0
Kd_y_max = 2 * Kd_y_static

Kp_z = np.zeros((N + 1, 1))
Kp_z_min = 0
Kp_z_max = 2 * Kp_z_static
Ki_z = np.zeros((N + 1, 1))
Ki_z_min = 0
Ki_z_max = 2 * Ki_z_static
Kd_z = np.zeros((N + 1, 1))
Kd_z_min = 0
Kd_z_max = 2 * Kd_z_static

Kp_psi = np.zeros((N + 1, 1))
Kp_psi_min = 0
Kp_psi_max = 2 * Kp_psi_static
Ki_psi = np.zeros((N + 1, 1))
Ki_psi_min = 0
Ki_psi_max = 2 * Ki_psi_static
Kd_psi = np.zeros((N + 1, 1))
Kd_psi_min = 0
Kd_psi_max = 2 * Kd_psi_static

Kp_phi = np.zeros((N + 1, 1))
Kp_phi_min = 0
Kp_phi_max = 2 * Kp_phi_static
Ki_phi = np.zeros((N + 1, 1))
Ki_phi_min = 0
Ki_phi_max = 2 * Ki_phi_static
Kd_phi = np.zeros((N + 1, 1))
Kd_phi_min = 0
Kd_phi_max = 2 * Kd_phi_static

Kp_tta = np.zeros((N + 1, 1))
Kp_tta_min = 0
Kp_tta_max = 2 * Kp_tta_static
Ki_tta = np.zeros((N + 1, 1))
Ki_tta_min = 0
Ki_tta_max = 2 * Ki_tta_static
Kd_tta = np.zeros((N + 1, 1))
Kd_tta_min = 0
Kd_tta_max = 2 * Kd_tta_static

e1 = np.zeros((N + 1, n))
e2 = np.zeros((N + 1, n))

# ---------------------   Initialize Control inputs  ----------------------------------------#
u1 = np.zeros((N + 1, 1))
u2 = np.zeros((N + 1, 1))
u3 = np.zeros((N + 1, 1))
u4 = np.zeros((N + 1, 1))
tta_d = np.zeros((N + 1, 1))
phi_d = np.zeros((N + 1, 1))

# Reset Integrator terms
ei_psi = 0
ei_phi = 0
ei_tta = 0
ei_z = 0
ei_x = 0
ei_y = 0

# %%
K_z_static = torch.tensor([[Kp_z_static, Ki_z_static, Kd_z_static]], dtype=torch.float32)
K_phi_static = torch.tensor([[Kp_phi_static, Ki_phi_static, Kd_phi_static]], dtype=torch.float32)
K_tta_static = torch.tensor([[Kp_tta_static, Ki_tta_static, Kd_tta_static]], dtype=torch.float32)
K_psi_static = torch.tensor([[Kp_psi_static, Ki_psi_static, Kd_psi_static]], dtype=torch.float32)

K_z_domain = 0.5 * K_z_static
K_phi_domain = 0.5 * torch.tensor([[Kp_phi_static, 0.1, Kd_phi_static]], dtype=torch.float32)  # 0.5*K_phi_static
K_tta_domain = 0.5 * torch.tensor([[Kp_tta_static, 0.1, Kd_tta_static]], dtype=torch.float32)  # 0.5*K_tta_static
K_psi_domain = 0.5 * torch.tensor([[Kp_psi_static, 0.1, Kd_psi_static]], dtype=torch.float32)  # 0.5*K_psi_static

# %% Actor & Critic Structures
print("Create Network structures ...\n")
# ----------------------------- Z Actor Critic ------------------------------------------------------#
z_agent = NewAgent(state_size=7, hidden_size=5, alpha=0.01, gamma=0.99, K_static=K_z_static, K_domain=K_z_domain,
                   Sigma_domain=Sigma_domain_z)
z_agent = reset_params(z_agent, init_type="uniform")

# ----------------------------- Phi Actor Critic ------------------------------------------------------#
phi_agent = NewAgent(state_size=7, hidden_size=5, alpha=0.0001, gamma=0.99, K_static=K_phi_static,
                     K_domain=K_phi_domain, Sigma_domain=Sigma_domain_phi)
phi_agent = reset_params(phi_agent, init_type="uniform")

# ----------------------------- Tta Actor Critic ------------------------------------------------------#
tta_agent = NewAgent(state_size=7, hidden_size=5, alpha=0.0001, gamma=0.99, K_static=K_tta_static,
                     K_domain=K_tta_domain, Sigma_domain=Sigma_domain_tta)
tta_agent = reset_params(tta_agent, init_type="uniform")

# ----------------------------- Psi Actor Critic ------------------------------------------------------#
psi_agent = NewAgent(state_size=7, hidden_size=5, alpha=0.001, gamma=0.99, K_static=K_psi_static, K_domain=K_psi_domain,
                     Sigma_domain=Sigma_domain_psi)
psi_agent = reset_params(psi_agent, init_type="uniform")

# %%-------------- additional params ------------------
z_m = np.array([])
psi_m = np.array([])
phi_m = np.array([])
tta_m = np.array([])

mu_z = np.array([])
mu_psi = np.array([])
mu_phi = np.array([])
mu_tta = np.array([])

sigma_z = np.array([])
sigma_psi = np.array([])
sigma_phi = np.array([])
sigma_tta = np.array([])

# ---------------- Initiate loss functions ---------------#
z_agent_losses = np.array([])
phi_agent_losses = np.array([])
tta_agent_losses = np.array([])
psi_agent_losses = np.array([])

# %%
# ---------------------------------------------  Main Loop  ----------------------------------------------#
print("Processing(Main Loop) ...\n")
ep_z = 0
ep_phi = 0
ep_tta = 0
ep_psi = 0

s_z = np.zeros((N + 1, 1))
s_phi = np.zeros((N + 1, 1))
s_tta = np.zeros((N + 1, 1))
s_psi = np.zeros((N + 1, 1))

p = int(Tf / dt)
for k in range(N):

    # ---------------------  Add Uncertainty to model  ----------------------------#
    # if Uncertainty==1:
    #     if k > int(Tf/3/dt):
    #         m = 0.65*2 
    #     if k > int(2*Tf/3/dt):
    #         m = 0.65*3
    m = m0
    if Uncertainty == 1:
        m = m0 * (0 <= k < int(0.2 * p)) + 2 * m0 * (int(0.2 * p) <= k < int(0.4 * p)) + \
            3 * m0 * (int(0.4 * p) <= k < int(0.6 * p)) + 2 * m0 * (int(0.6 * p) <= k < int(0.8 * p)) + \
            m0 * (int(0.8 * p) <= k < int(1.0 * p) + 1)

    M[k, :] = m
    ## ---------  Sliding Mode Control
    # [us, Uxy] = SMC(t[k], x, y, Uxy, m)
    # u = us

    # Self-tuning PID Control
    u[0] = u1[k]
    u[1] = u2[k]
    u[2] = u3[k]
    u[3] = u4[k]

    # Dynamic + Noise
    k1 = Quadcopter_Dynamics_with_Disturbance(0, x, u, m, dist, k)
    k2 = Quadcopter_Dynamics_with_Disturbance(0, x + dt / 2 * k1, u, m, dist, k)
    k3 = Quadcopter_Dynamics_with_Disturbance(0, x + dt / 2 * k2, u, m, dist, k)
    k4 = Quadcopter_Dynamics_with_Disturbance(0, x + dt * k3, u, m, dist, k)

    x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    y = Quadcopter_Dynamics_with_Disturbance(0, x, u, m, dist, k)
    y = y[0::2]

    # Add Noise to outputs
    # x[4]   = Add_Noise(x[4], "Angular")
    # x[6:12:2] = Add_Noise(x[6:12:2], "Linear")

    Xs[k + 1, :] = x.T
    U = np.append(U, u.T, axis=0)

    # Desired X, Y
    Xd[k + 1, 0] = phi_d[k]
    Xd[k + 1, 1] = 0
    Xd[k + 1, 2] = tta_d[k]
    Xd[k + 1, 3] = 0
    if k > 0:
        Xd[k + 1, 1] = (phi_d[k] - phi_d[k - 1]) / dt
        Xd[k + 1, 3] = (tta_d[k] - tta_d[k - 1]) / dt

    ## Control Error
    e2[k + 1, :] = Xd[k + 1, :] - x.T

    # ------------------------------------    Config PID Errors   -------------------------------------------#
    ep_phi = e2[k, 0]
    ei_phi = ei_phi + ep_phi * dt
    ed_phi = e2[k, 1]

    ep_tta = e2[k, 2]
    ei_tta = ei_tta + ep_tta * dt
    ed_tta = e2[k, 3]

    ep_psi = e2[k, 4]
    ei_psi = ei_psi + ep_psi * dt
    ed_psi = e2[k, 5]

    ep_z = e2[k, 6]
    ei_z = ei_z + ep_z * dt
    ed_z = e2[k, 7]

    ep_x = e2[k, 8]
    ei_x = ei_x + ep_x * dt
    ed_x = e2[k, 9]

    ep_y = e2[k, 10]
    ei_y = ei_y + ep_y * dt
    ed_y = e2[k, 11]

    # %% Adaptive gains using Actor

    # ----------------------  Z network ----------------------
    z_state1 = torch.tensor([[u1[k][0], u1[k - 1][0], ep_z, ei_z, ed_z, Xs[k, 6], Xs[k - 1, 6]]], dtype=torch.float32)
    z_state2 = torch.tensor([[Xs[k, 6], Xs[k - 1, 6]]], dtype=torch.float32)
    z_target = Xs[k + 1, 6] * torch.ones((1, 1))

    z_agent.actor_critic.u.weight.data = torch.tensor([[ep_z, ei_z, ed_z]], dtype=torch.float32)
    z_output = z_agent.choose_action(z_state1, z_state2, z_target)
    z_m = np.append(z_m, z_output)

    # Extract dynamic gains    
    PID_gains_z = z_agent.actor_critic.PID()
    Kp_z[k + 1] = PID_gains_z[0][0].data.numpy()
    Ki_z[k + 1] = PID_gains_z[0][1].data.numpy()
    Kd_z[k + 1] = PID_gains_z[0][2].data.numpy()

    # -------------------------  PHI network -------------------------

    phi_state1 = torch.tensor([[u2[k][0], u2[k - 1][0], ep_phi, ei_phi, ed_phi, Xs[k, 0], Xs[k - 1, 0]]],
                              dtype=torch.float32)
    phi_state2 = torch.tensor([[Xs[k, 0], Xs[k - 1, 0]]], dtype=torch.float32)
    phi_target = Xs[k + 1, 0] * torch.ones((1, 1))

    phi_agent.actor_critic.u.weight.data = torch.tensor([[ep_phi, ei_phi, ed_phi]], dtype=torch.float32)
    phi_output = phi_agent.choose_action(phi_state1, phi_state2, phi_target)
    phi_m = np.append(phi_m, phi_output)

    # Extract dynamic gains    
    PID_gains_phi = phi_agent.actor_critic.PID()
    Kp_phi[k + 1] = PID_gains_phi[0][0].data.numpy()
    Ki_phi[k + 1] = PID_gains_phi[0][1].data.numpy()
    Kd_phi[k + 1] = PID_gains_phi[0][2].data.numpy()

    # ------------------------- TTA network ------------------------

    tta_state1 = torch.tensor([[u3[k][0], u3[k - 1][0], ep_tta, ei_tta, ed_tta, Xs[k, 2], Xs[k - 1, 2]]],
                              dtype=torch.float32)
    tta_state2 = torch.tensor([[Xs[k, 2], Xs[k - 1, 2]]], dtype=torch.float32)
    tta_target = Xs[k + 1, 2] * torch.ones((1, 1))

    tta_agent.actor_critic.u.weight.data = torch.tensor([[ep_tta, ei_tta, ed_tta]], dtype=torch.float32)
    tta_output = tta_agent.choose_action(tta_state1, tta_state2, tta_target)
    tta_m = np.append(tta_m, tta_output)

    # Extract dynamic gains   
    PID_gains_tta = tta_agent.actor_critic.PID()
    Kp_tta[k + 1] = PID_gains_tta[0][0].data.numpy()
    Ki_tta[k + 1] = PID_gains_tta[0][1].data.numpy()
    Kd_tta[k + 1] = PID_gains_tta[0][2].data.numpy()

    # -------------------------  PSI network ---------------------

    psi_state1 = torch.tensor([[u4[k][0], u4[k - 1][0], ep_psi, ei_psi, ed_psi, Xs[k, 4], Xs[k - 1, 4]]],
                              dtype=torch.float32)
    psi_state2 = torch.tensor([[Xs[k, 4], Xs[k - 1, 4]]], dtype=torch.float32)
    psi_target = Xs[k + 1, 4] * torch.ones((1, 1))

    psi_agent.actor_critic.u.weight.data = torch.tensor([[ep_psi, ei_psi, ed_psi]], dtype=torch.float32)
    psi_output = psi_agent.choose_action(psi_state1, psi_state2, psi_target)
    psi_m = np.append(psi_m, psi_output)

    # Extract dynamic gains    
    PID_gains_psi = psi_agent.actor_critic.PID()
    Kp_psi[k + 1] = PID_gains_psi[0][0].data.numpy()
    Ki_psi[k + 1] = PID_gains_psi[0][1].data.numpy()
    Kd_psi[k + 1] = PID_gains_psi[0][2].data.numpy()

    # ------------------------------------------- Adaptive PID Control   ------------------------------------#
    Kp_x[k + 1] = Kp_x_static
    Ki_x[k + 1] = Ki_x_static
    Kd_x[k + 1] = Kd_x_static
    Kp_y[k + 1] = Kp_y_static
    Ki_y[k + 1] = Ki_y_static
    Kd_y[k + 1] = Kd_y_static

    # ------------------  Update u(t)   --------------------------------------------#
    # U1 = z_agent.actor_critic.CtrlSignal()
    # u1[k+1] = U1[0][0].data.numpy()
    # uz = Kp_x[k+1]*ep_x + Ki_x[k+1]*ei_x + Kd_x[k+1]*ed_x
    # print(uz-u1[k+1])

    # U2 = phi_agent.actor_critic.CtrlSignal()
    # u2[k+1] = U2[0][0].data.numpy()

    # U3 = tta_agent.actor_critic.CtrlSignal()
    # u3[k+1] = U3[0][0].data.numpy()

    # U4 = psi_agent.actor_critic.CtrlSignal()
    # u4[k+1] = U4[0][0].data.numpy()

    u1[k + 1] = Kp_z[k + 1] * ep_z + Ki_z[k + 1] * ei_z + Kd_z[k + 1] * ed_z

    u2[k + 1] = Kp_phi[k + 1] * ep_phi + Ki_phi[k + 1] * ei_phi + Kd_phi[k + 1] * ed_phi

    u3[k + 1] = Kp_tta[k + 1] * ep_tta + Ki_tta[k + 1] * ei_tta + Kd_tta[k + 1] * ed_tta

    u4[k + 1] = Kp_psi[k + 1] * ep_psi + Ki_psi[k + 1] * ei_psi + Kd_psi[k + 1] * ed_psi

    tta_d[k + 1] = Kp_x[k + 1] * ep_x + Ki_x[k + 1] * ei_x + Kd_x[k + 1] * ed_x

    phi_d[k + 1] = -(Kp_y[k + 1] * ep_y + Ki_y[k + 1] * ei_y + Kd_y[k + 1] * ed_y)

    # -------------  Learn Actor & Critic  ------------------------------------#
    # ------- Z ------
    prediction_ = torch.tensor(z_output, dtype=torch.float32)
    reward = -(prediction_ - z_target) ** 2
    R_z[k, :] = reward.data.numpy()[0][0]

    next_z_state1 = torch.tensor([[u1[k + 1][0], u1[k][0], ep_z, ei_z, ed_z, Xs[k + 1, 6], Xs[k, 6]]],
                                 dtype=torch.float32)
    next_z_state2 = torch.tensor([[Xs[k + 1, 6], Xs[k, 6]]], dtype=torch.float32)

    z_agent.learn(z_state1, z_state2, reward, next_z_state1, next_z_state2)

    z_agent_losses = np.append(z_agent_losses, z_agent.Total_Loss.data.numpy()[0][0])

    # ------- phi ------
    prediction_ = torch.tensor(phi_output, dtype=torch.float32)
    reward = -(prediction_ - phi_target) ** 2 - (prediction_ - Xd[k + 1, 0] * torch.ones((1, 1))) ** 2
    R_phi[k, :] = reward.data.numpy()[0][0]

    next_phi_state1 = torch.tensor([[u2[k + 1][0], u2[k][0], ep_phi, ei_phi, ed_phi, Xs[k + 1, 0], Xs[k, 0]]],
                                   dtype=torch.float32)
    next_phi_state2 = torch.tensor([[Xs[k + 1, 0], Xs[k, 0]]], dtype=torch.float32)

    phi_agent.learn(phi_state1, phi_state2, reward, next_phi_state1, next_phi_state2)
    phi_agent_losses = np.append(phi_agent_losses, phi_agent.Total_Loss.data.numpy()[0][0])

    # ------- tta ------
    prediction_ = torch.tensor(tta_output, dtype=torch.float32)
    reward = -(prediction_ - tta_target) ** 2
    R_tta[k, :] = reward.data.numpy()[0][0]

    next_tta_state1 = torch.tensor([[u3[k + 1][0], u3[k][0], ep_tta, ei_tta, ed_tta, Xs[k + 1, 2], Xs[k, 2]]],
                                   dtype=torch.float32)
    next_tta_state2 = torch.tensor([[Xs[k + 1, 2], Xs[k, 2]]], dtype=torch.float32)

    tta_agent.learn(tta_state1, tta_state2, reward, next_tta_state1, next_tta_state2)
    tta_agent_losses = np.append(tta_agent_losses, tta_agent.Total_Loss.data.numpy()[0][0])

    # ------- psi ------
    prediction_ = torch.tensor(psi_output, dtype=torch.float32)
    reward = -(prediction_ - psi_target) ** 2
    R_psi[k, :] = reward.data.numpy()[0][0]

    next_psi_state1 = torch.tensor([[u4[k + 1][0], u4[k][0], ep_psi, ei_psi, ed_psi, Xs[k + 1, 4], Xs[k, 4]]],
                                   dtype=torch.float32)
    next_psi_state2 = torch.tensor([[Xs[k + 1, 4], Xs[k, 4]]], dtype=torch.float32)

    psi_agent.learn(psi_state1, psi_state2, reward, next_psi_state1, next_psi_state2)
    psi_agent_losses = np.append(psi_agent_losses, psi_agent.Total_Loss.data.numpy()[0][0])

    # ------------ Extract Average and variance -----------------
    mu_z = np.append(mu_z, z_agent.actor_critic.Mu.data.numpy()[0][0])
    sigma_z = np.append(sigma_z, z_agent.actor_critic.Sigma.data.numpy()[0][0])

    mu_phi = np.append(mu_phi, phi_agent.actor_critic.Mu.data.numpy()[0][0])
    sigma_phi = np.append(sigma_phi, phi_agent.actor_critic.Sigma.data.numpy()[0][0])

    mu_tta = np.append(mu_tta, tta_agent.actor_critic.Mu.data.numpy()[0][0])
    sigma_tta = np.append(sigma_tta, tta_agent.actor_critic.Sigma.data.numpy()[0][0])

    mu_psi = np.append(mu_psi, psi_agent.actor_critic.Mu.data.numpy()[0][0])
    sigma_psi = np.append(sigma_psi, psi_agent.actor_critic.Sigma.data.numpy()[0][0])

    # --------------------------------------    Security   ---------------------------------------------#
    if isnan(x[0]):
        raise Exception("Oops!, NaN occured!!!  -----------------------------------")

    if abs(x[0]) > np.deg2rad(45):
        raise Exception("Output went out of bound.")

    if int(k * dt) % 10 == 0:
        text = f"t = {int(k * dt)}"
        sys.stdout.write("\r" + text)

Xs = Xs[1:, :]

# %%
z_upper = mu_z + 3 * sigma_z
z_lower = mu_z - 3 * sigma_z
phi_upper = np.rad2deg(mu_phi + 3 * sigma_phi)
phi_lower = np.rad2deg(mu_phi - 3 * sigma_phi)
tta_upper = np.rad2deg(mu_tta + 3 * sigma_tta)
tta_lower = np.rad2deg(mu_tta - 3 * sigma_tta)
psi_upper = np.rad2deg(mu_psi + 3 * sigma_psi)
psi_lower = np.rad2deg(mu_psi - 3 * sigma_psi)

# %%
# ------------------------------------------   Plot Results    -------------------------------------------#
print("\n")
print("plot results ...\n\n")
plt.figure()
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 18})
plt.tight_layout()

plt.title("Linear Positions")
plt.plot(t, Xd[:-1, 8], 'g--', label=r"$X_d$")
plt.plot(t, Xs[:, 8], label="X", color='b', linewidth=2)
plt.plot(t, Xd[:-1, 10], 'y--', label=r"$Y_d$")
plt.plot(t, Xs[:, 10], label="Y", color='r', linewidth=2)
plt.xlabel('Time(sec)')
plt.ylabel("Amp(m)")
plt.legend(loc='best')

# plt.savefig('Position_xy_constant5', dpi=1200)
plt.show()

# %%--------------------  Altitude ----------------------------------------#
plt.figure()
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 18})
plt.tight_layout()

plt1 = plt.subplot2grid((12, 1), (0, 0), rowspan=6, colspan=1)
plt2 = plt.subplot2grid((12, 1), (6, 0), rowspan=6, colspan=1)

plt1.set_title("Altitude")
plt1.plot(t, Xd[0:-1, 6], 'g--', label=r"$Z_d$", linewidth=2)
# plt1.plot(t, z_upper, t, z_lower , color='grey', linewidth=0.5)
plt1.fill_between(t, z_lower, z_upper, color='silver')
plt1.plot(t, z_m, label=r"$Z_m$", color='grey', linewidth=2)
plt1.plot(t, Xs[:, 6], label="Z", color='b', linewidth=3)
plt1.set_xticklabels([])
plt1.set_ylabel("Amp(m)")

# plt2.set_title("PID Gains")
plt2.plot(t, Kp_z[1:] - Kp_z_static, label=r"$K_p$", color='b', linewidth=2)
plt2.plot(t, Ki_z[1:] - Ki_z_static, label=r"$K_i$", color='r', linewidth=2)
plt2.plot(t, Kd_z[1:] - Kd_z_static, label=r"$K_d$", color='g', linewidth=2)
plt2.set_xlabel('Time(sec)')
plt2.set_ylabel("PID gains")
plt1.legend(loc="lower right")
plt2.legend(loc="upper right")

# plt.savefig('Altitude_constant6', dpi=1200)
plt.show()

# --------------------  Attitudes -------------------------------------------#
# %%--------------------  phi ----------------------#
plt.figure()
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 18})

plt1 = plt.subplot2grid((12, 1), (0, 0), rowspan=6, colspan=1)
plt2 = plt.subplot2grid((12, 1), (6, 0), rowspan=6, colspan=1)

plt1.set_title("Roll Angle")
# plt1.plot(t, phi_lower, t, phi_upper, color='silver', linewidth=0.5)
plt1.fill_between(t, phi_lower, phi_upper, color='silver')
plt1.plot(t, np.rad2deg(phi_m), label=r"$\phi_m$", color='grey', linewidth=2)
plt1.plot(t, np.rad2deg(Xd[0:-1, 0]), 'g--', label=r"$\phi_d$", linewidth=2)
plt1.plot(t, np.rad2deg(Xs[:, 0]), label=r"$\phi$", color='b', linewidth=3)
plt1.set_xticklabels([])
plt1.set_ylabel("Angle(deg)")

# plt2.set_title("PID Gains")
plt2.plot(t, Kp_phi[1:] - Kp_phi_static, label=r"$K_p$", color='b', linewidth=2)
plt2.plot(t, Ki_phi[1:] - Ki_phi_static, label=r"$K_i$", color='r', linewidth=2)
plt2.plot(t, Kd_phi[1:] - Kd_phi_static, label=r"$K_d$", color='g', linewidth=2)
plt2.set_xlabel('Time(sec)')
plt2.set_ylabel("PID gains")
plt1.legend(loc="upper right")
plt2.legend(loc="upper right")

# plt.savefig('roll_constant4', dpi=1200)
plt.show()

# %%--------------------  tta ----------------------#
plt.figure()
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 18})
plt.tight_layout()

plt1 = plt.subplot2grid((12, 1), (0, 0), rowspan=6, colspan=1)
plt2 = plt.subplot2grid((12, 1), (6, 0), rowspan=6, colspan=1)

plt1.set_title("Pitch Angle")
# plt1.plot(t,tta_upper, t, tta_lower, color='grey', linewidth=0.5) 
plt1.fill_between(t, tta_lower, tta_upper, color='silver')
plt1.plot(t, np.rad2deg(tta_m), label=r"$\theta_m$", color='grey', linewidth=2)
plt1.plot(t, np.rad2deg(Xd[0:-1, 2]), 'g--', label=r"$\theta_d$", linewidth=2)
plt1.plot(t, np.rad2deg(Xs[:, 2]), label=r"$\theta$", color='b', linewidth=3)
plt1.set_xticklabels([])
plt1.set_ylabel("Angle(deg)")

# plt2.set_title("PID Gains")
plt2.plot(t, Kp_tta[1:] - Kp_tta_static, label=r"$K_p$", color='b', linewidth=2)
plt2.plot(t, Ki_tta[1:] - Ki_tta_static, label=r"$K_i$", color='r', linewidth=2)
plt2.plot(t, Kd_tta[1:] - Kd_tta_static, label=r"$K_d$", color='g', linewidth=2)
plt2.set_xlabel('Time(sec)')
plt2.set_ylabel("PID gains")
plt1.legend(loc="lower right")
plt2.legend(loc="upper right")

# plt.savefig('pitch_constant6', dpi=1200)
plt.show()

# %%--------------------  psi ----------------------#
plt.figure()
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 18})
plt.tight_layout()

plt1 = plt.subplot2grid((12, 1), (0, 0), rowspan=6, colspan=1)
plt2 = plt.subplot2grid((12, 1), (6, 0), rowspan=6, colspan=1)

plt1.set_title("Yaw Angle")
# plt1.plot(t, psi_upper, t, psi_lower, color='grey', linewidth=0.5)
plt1.fill_between(t, psi_upper, psi_lower, color='silver')
plt1.plot(t, np.rad2deg(psi_m), label=r"$\psi_m$", color='grey', linewidth=2)
plt1.plot(t, np.rad2deg(Xd[0:-1, 4]), 'g--', label=r"$\psi_d$", linewidth=2)
plt1.plot(t, np.rad2deg(Xs[:, 4]), label=r"$\psi$", color='b', linewidth=3)
plt1.set_xticklabels([])
plt1.set_ylabel("Angle(deg)")

# plt2.set_title("PID Gains")
plt2.plot(t, Kp_psi[1:] - Kp_psi_static, label=r'$K_p$', color='b', linewidth=2)
plt2.plot(t, Ki_psi[1:] - Ki_psi_static, label=r'$K_i$', color='r', linewidth=2)
plt2.plot(t, Kd_psi[1:] - Kd_psi_static, label=r'$K_d$', color='g', linewidth=2)
plt2.set_xlabel('Time(sec)')
plt2.set_ylabel("PID gains")
plt1.legend(loc="lower right")
plt2.legend(loc="upper right")

# plt.savefig('yaw_constant3', dpi=1200)
plt.show()

# %%--------------------  Guassian-Markov Disturbances ----------------------#
plt.figure()
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 18})
plt.tight_layout()

plt.title("Guassian-Morkov Disturbances")
plt.plot(t, Xs[:, 12], label=r'$d_1$', color='b', linewidth=2)
plt.plot(t, Xs[:, 13], label=r'$d_2$', color='r', linewidth=2)
plt.plot(t, Xs[:, 14], label=r'$d_3$', color='g', linewidth=2)
plt.xlabel("Time(sec)")
plt.ylabel("Amp(1/s^2)")
plt.legend(loc="lower right")

# plt.savefig('Loss_constant1', dpi=1200)
plt.show()

# %%--------------------  Rewards ----------------------#
plt.figure()
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 18})

plt1 = plt.subplot2grid((12, 1), (0, 0), rowspan=3, colspan=1)
plt2 = plt.subplot2grid((12, 1), (3, 0), rowspan=3, colspan=1)
plt3 = plt.subplot2grid((12, 1), (6, 0), rowspan=3, colspan=1)
plt4 = plt.subplot2grid((12, 1), (9, 0), rowspan=3, colspan=1)

plt1.set_title("Rewards")
plt1.plot(t, R_z[:-1], label=r'$R_z$', color='b', linewidth=2)
plt1.set_xticklabels([])

plt2.plot(t, R_phi[:-1], label=r'$R_\phi$', color='r', linewidth=2)
plt2.set_xticklabels([])

plt3.plot(t, R_tta[:-1], label=r'$R_\theta$', color='g', linewidth=2)
plt3.set_xticklabels([])

plt4.plot(t, R_psi[:-1], label=r'$R_\psi$', color='y', linewidth=2)
plt4.set_xlabel('Time(sec)')

plt1.legend(loc="best")
plt2.legend(loc="best")
plt3.legend(loc="best")
plt4.legend(loc="best")

# plt.savefig('R_constant1', dpi=1200)
plt.show()

# %%--------------------  Loss Functions ----------------------#
plt.figure()
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 18})

plt1 = plt.subplot2grid((12, 1), (0, 0), rowspan=3, colspan=1)
plt2 = plt.subplot2grid((12, 1), (3, 0), rowspan=3, colspan=1)
plt3 = plt.subplot2grid((12, 1), (6, 0), rowspan=3, colspan=1)
plt4 = plt.subplot2grid((12, 1), (9, 0), rowspan=3, colspan=1)

plt1.set_title("Loss Functions")
plt1.plot(t, z_agent_losses, label=r'$L_z$', color='b', linewidth=2)
plt1.set_xticklabels([])

plt2.plot(t, phi_agent_losses, label=r'$L_\phi$', color='r', linewidth=2)
plt2.set_xticklabels([])

plt3.plot(t, tta_agent_losses, label=r'$L_\theta$', color='g', linewidth=2)
plt3.set_xticklabels([])

plt4.plot(t, psi_agent_losses, label=r'$L_\psi$', color='y', linewidth=2)
plt4.set_xlabel('Time(sec)')

plt1.legend(loc="best")
plt2.legend(loc="best")
plt3.legend(loc="best")
plt4.legend(loc="best")

# plt.savefig('Loss_constant1', dpi=1200)
plt.show()

# %% Save Data
# from save_data import save_data
# self_tuning = 1
# if self_tuning == 1:
#     filename = "saved_data\\A2C_PID_data_with_square_path"
# else:
#     filename = "saved_data\\PID_data_with_Guassian_Dist_seed2"

# # with open(filename+"1.npy", 'wb') as f:
# #     np.save(f,Xs)

# Save_data(filename+"7th_bahman.xls", t, Xs, Xd, M)

# %% End
print("done!")
