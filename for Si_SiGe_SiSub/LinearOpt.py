# refractive

import numpy as np
import math
def Fresnel_SH(OptPar,PLayer,Kin):
    # PLayer [angle,bonds,xyz]

    # Optical     
    N1 = OptPar['RefInd_w']
    N2 = OptPar['RefInd_2w']
    
    # Parameters
    noLayers = len(N1)
    thetaO = [None] * noLayers
    Kout = [None] * noLayers
    
    # Calculate SHG from which layer, Ex: for air/SiO2/Si, Start = 2 -> only from Si
    Start = OptPar['Start']  

    # sin(theta_final)
    Kin_final = Kin[-1][0]
    thetaO[-1] = np.arcsin(N1[-1]/N2[-1]*Kin_final)
    
    # Kout has same order as Kin  Ex:[air, SiO2, Si ]
    Kout[-1] = [np.sin(thetaO[-1]), 0, np.cos(thetaO[-1])]
    
    # noLayers-2 : 倒數第二個
    for Li in np.arange(noLayers-2,-1,-1):
        thetaO[Li] = np.arcsin(N2[Li+1]/N2[Li]*np.sin(thetaO[Li+1]))
        # 2w K-vector (wave vector) direction
        Kout[Li] = [np.sin(thetaO[Li]), 0, np.cos(thetaO[Li])]
  
    # Analyzer [xyz/xyz]
    PhiOut = OptPar['PolOutDeg']/180*np.pi
    EInterfPol = [None] * noLayers
    EInterfEleFF = [None] * noLayers
    EInterfElePol = [None] * noLayers
    AnaMetrix = [None] * noLayers
    
    for Li in np.arange(Start,noLayers):
        AnaMetrix[Li] = Analyzer_Matrix_on_source(thetaO[Li], PhiOut)
        
        # Far Field 
        # Equation 8 of 10.1088/2040-8978/18/3/035501
        EInterfEleFF[Li] = np.matmul(
            np.eye(3)-np.kron([Kout[Li]],np.transpose([Kout[Li]])),
            np.transpose(PLayer[Li],(0,2,1)))
 
        # Analyzed bond-wise E-field [phi,xyz,# bonds] for PMT 前的偏振片 p or s
        EInterfElePol[Li] = np.matmul(AnaMetrix[Li],EInterfEleFF[Li])
        
        # phase different for different layer
        # PLayer[Li] = PLayer[Li]*(np.cos(4*np.cos(thetaO[Li])*np.pi*d[Li]/522)+1j*np.sin(4*np.cos(thetaO[Li])*np.pi*d[Li]/522))
        
        # Analyzed E-field [phi,xyz] 把bond加在一起
        EInterfPol[Li] = np.sum(EInterfElePol[Li],axis=2)
    
    # Analyzed Intensity [phi]
    EInterfPol_total = 0
    for Li in np.arange(Start,noLayers):
        EInterfPol_total = EInterfPol_total + EInterfPol[Li]
        
    Isum = np.sum(EInterfPol_total*np.conj(EInterfPol_total),axis=1).real
    
    return Isum 

def Analyzer_Matrix_on_source(thetaO,PhiOut):
   
    Ana = [[-np.cos(thetaO)*np.sin(PhiOut),
           np.cos(PhiOut),
           np.sin(thetaO)*np.sin(PhiOut)]]
    AnaMetrix = np.kron(Ana, np.transpose(Ana))
    
    return AnaMetrix
    