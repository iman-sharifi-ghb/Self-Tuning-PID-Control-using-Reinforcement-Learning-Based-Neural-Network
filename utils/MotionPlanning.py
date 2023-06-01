import numpy as np
from scipy import signal
global Tf
Tf = 100

def CreateDesiredTrajectory(t):
    
    global Tf
    # ------- Static point
    Xd = 0;    Yd = 0;    Zd = 0;    
    dXd = 0;    dYd = 0;    dZd = 0;    
    ddXd = 0;    ddYd = 0;    ddZd = 0;
    
    phid = 0;   Ttad = 0;   Psid = 0;    
    dphid = 0;  dTtad = 0;  dPsid = 0;    
    ddphid = 0; ddTtad = 0; ddPsid = 0;
    
    ## -------- Square path
    # Xd = 1+signal.square(4 * np.pi / Tf * t)  ;
    # Yd = 1+signal.square(4 * np.pi / Tf * (t- Tf/8 ));
    # Zd = 1;    
    # dXd = 0;    dYd = 0;    dZd = 0;    
    # ddXd = 0;    ddYd = 0;    ddZd = 0;
    
    # phid = 0;   Ttad = 0;   Psid = 0;    
    # dphid = 0;  dTtad = 0;  dPsid = 0;    
    # ddphid = 0; ddTtad = 0; ddPsid = 0;
    
    ## -------- Helical Path
    # r = 1; w = 4*np.pi/Tf;a = np.deg2rad(5);
    # Xd = r*np.cos(w*t);
    # Yd = r*np.sin(w*t);
    # Zd = 0.05*t;
    
    # dXd = -r*w*np.sin(w*t);
    # dYd = r*w*np.cos(w*t);
    # dZd = 0.05;
    
    # ddXd = -r*w**2*np.cos(w*t);
    # ddYd = -r*w**2*np.sin(w*t);
    # ddZd = 0;
    
    phid = 0;    Ttad = 0;    Psid = 0;    
    dphid = 0;    dTtad = 0;    dPsid = 0;    
    ddphid = 0;    ddTtad = 0;    ddPsid = 0;
    
    DesAttitude = np.array([phid,dphid,ddphid,Ttad,dTtad,ddTtad,Psid,dPsid,ddPsid]);
    DesPosition = np.array([Zd,dZd,ddZd,Xd,dXd,ddXd,Yd,dYd,ddYd]);
    
    return DesPosition, DesAttitude