import numpy as np
# Dynamic Params
global m ,g
global l ,b, d
global Jx, Jy, Jz
global a1, a2, a3
global b1 ,b2, b3

m = 0.65;
g = 9.81;
l = 0.23;
Jx = 7.5e-3;
Jy = 7.5e-3;
Jz = 1.3e-2;
Jr = 1e-5;
b = 3.13e-5;
d = 7.5e-5;

a1 = (Jy-Jz)/Jx;
a2 = (Jz-Jx)/Jy;
a3 = (Jx-Jy)/Jz;

b1 = l/Jx;
b2 = l/Jy;
b3 = 1/Jz;

def Quadcopter_Dynamics(t, x, u, m):

    x_dot = np.zeros((12,1))
    
    Ux = np.cos(x[0])*np.sin(x[2])*np.cos(x[4])+np.sin(x[0])*np.sin(x[4])
    Uy = np.cos(x[0])*np.sin(x[2])*np.sin(x[4])-np.sin(x[0])*np.cos(x[4])
    
    x_dot[0] = x[1]
    x_dot[1] = x[3]*x[5]*a1+u[1]*b1
    
    x_dot[2] = x[3]
    x_dot[3] = x[1]*x[5]*a2+u[2]*b2
    
    x_dot[4] = x[5]
    x_dot[5] = x[1]*x[3]*a3+u[3]*b3
    
    x_dot[6] = x[7]
    x_dot[7] = u[0]/m*(np.cos([0])*np.cos(x[2]))-g
    
    x_dot[8] = x[9]
    x_dot[9] = u[0]/m*Ux
    
    x_dot[10] = x[11]
    x_dot[11] = u[0]/m*Uy
    
    return x_dot

def Add_Noise(Input, Type="Angular"):
    
    if Type=="Angular":
        error_bond = 0.02
        Output = Input + np.deg2rad(error_bond)*(np.random.rand(Input.size,1)-0.5)
        
    if Type=="Linear":
        error_bond = 0.1
        Output = Input + error_bond*(np.random.rand(Input.size,1)-0.5)
            
    return Output


def Quadcopter_Dynamics_with_Disturbance(t, x, u, m, dist, k):

    x_dot = np.zeros((15,1))
    # d1, d2, d3 = x[12], x[13], x[14]
    
    Ux = np.cos(x[0])*np.sin(x[2])*np.cos(x[4])+np.sin(x[0])*np.sin(x[4])
    Uy = np.cos(x[0])*np.sin(x[2])*np.sin(x[4])-np.sin(x[0])*np.cos(x[4])
    
    x_dot[0] = x[1]
    x_dot[1] = x[3]*x[5]*a1+Jr*x[3]+u[1]*b1+dist*x[12]
    
    x_dot[2] = x[3]
    x_dot[3] = x[1]*x[5]*a2-Jr*x[1]+u[2]*b2+dist*x[13]
    
    x_dot[4] = x[5]
    x_dot[5] = x[1]*x[3]*a3+u[3]*b3+dist*x[14]*5
    
    x_dot[6] = x[7]
    x_dot[7] = u[0]/m*(np.cos([0])*np.cos(x[2]))-g
    
    x_dot[8] = x[9]
    x_dot[9] = u[0]/m*Ux
    
    x_dot[10] = x[11]
    x_dot[11] = u[0]/m*Uy
    
    # Disturbance Equations
    tau_s = 3.2; rho_star = 200.0
    np.random.seed(k)
    B_w = np.eye(3,3); 
    q_w = np.random.rand(3,1)-0.5
    
    d_dot = -1/tau_s*d+rho_star*B_w@q_w
    
    x_dot[12:] = d_dot
    
    return x_dot