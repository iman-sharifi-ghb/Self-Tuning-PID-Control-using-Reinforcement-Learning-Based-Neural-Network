# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 13:22:34 2021

@author: Iman Sharifi
"""
import pandas as pd


def Save_data(filename, t, X, Xd, M):
    df = pd.DataFrame({"t":t, "X":X[:,8], "Y":X[:,10], "Z":X[:,6],\
                        "Phi":X[:,0], "Tta":X[:,2], "Psi":X[:,4],\
                        "Xd":Xd[:-1,8], "Yd":Xd[:-1,10], "Zd":Xd[:-1,6],\
                        "Phid":Xd[:-1,0], "Ttad":Xd[:-1,2], "Psid":Xd[:-1,4], "M":M[:-1,0]})
    
    df.to_excel(filename)