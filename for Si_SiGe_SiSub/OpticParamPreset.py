import numpy as np 

def OpticParamPreset(pol):
    OptPar = {
        # % polarization -- (s:0; p:90)
        # % Fundamental Beam & Sensor (incident & reflection) angle
        'ThetaKinDeg' : 45,
        'ThetaKoutDeg' : 45,
        # RefractiveIndex
        # from 10.1016/j.apsusc.2016.08.091
        'RefInd_w' : [1, 1.4589+0.00099737j,3.565108694+0j,3.6537744067+0.0082106875j,3.565108694+0j],                    # air / sio2 / si/ Si0.74Ge0.26 / Si_sub
        'RefInd_2w': [1, 1.4673+0.0020319j, 4.219164564+0.03673713j,4.200669376+0.07498804876j,4.219164564+0.03673713j],
        # Calculate SHG from which layer, Ex: for air/SiO2/Si, Start = 2 -> only from Si
        'Start' : 2,
        'thickness': [0,5,5,80,600000] # nm
        }
    
    if pol == 'ss':
        OptPar['PolInDeg'] = 0
        OptPar['PolOutDeg'] = 0
       
    elif pol == 'pp':
        OptPar['PolInDeg'] = 90
        OptPar['PolOutDeg'] = 90
    
    elif pol == 'sp':
        OptPar['PolInDeg'] = 0
        OptPar['PolOutDeg'] = 90
        
    elif pol == 'ps':
        OptPar['PolInDeg'] = 90
        OptPar['PolOutDeg'] = 0
        
    elif pol == 'test':
        OptPar['PolInDeg'] = 90
        OptPar['PolOutDeg'] = 45
        
   
    # Linear Optics 
    N = OptPar['RefInd_w']
    
    # Parameters
    noLayers = len(N)
    noInterf = len(N)-1
    thetaI = [None] * noLayers
    Kin = [0] * noLayers                                    # K-vector (wave vector) direction
    thetaI[0] = OptPar['ThetaKinDeg']/180*np.pi
    Kin[0] = [np.sin(thetaI[0]), 0, -np.cos(thetaI[0])]
    
    # Snell's law
    for Li in np.arange(1,noLayers):
        thetaI[Li] = np.arcsin(N[Li-1]/N[Li]*np.sin(thetaI[Li-1])) 
        Kin[Li] = [np.sin(thetaI[Li]), 0, -np.cos(thetaI[Li])]
     
    # Fresnel transmission coeff.: recursive method
    PhiIn = OptPar['PolInDeg']/180*np.pi
    Ein = [0] * noLayers
    
    # s-pol transmission
    ts = [1] * noInterf
    ts012 = [1] * noInterf
    # p-pol transmission
    tp = [1] * noInterf
    tp012 = [1] * noInterf
    
    # Equation 3 of 10.1088/2040-8978/18/3/035501
    Ein[0] = [np.cos(thetaI[0])*np.sin(PhiIn),
                np.cos(PhiIn),
                np.sin(thetaI[0])*np.sin(PhiIn)]
    
    # Parameter for absorption
    d = OptPar['thickness']
    absor_w = [1] * noLayers

    for Ii in np.arange(noInterf):
        # s-pol
        ts[Ii] = 2*N[Ii]*np.cos(thetaI[Ii]) / (N[Ii]*np.cos(thetaI[Ii])+N[Ii+1]*np.cos(thetaI[Ii+1]))
        ts012[Ii] = np.prod(ts)
        # p-pol
        tp[Ii] = 2*N[Ii]*np.cos(thetaI[Ii]) / (N[Ii+1]*np.cos(thetaI[Ii])+N[Ii]*np.cos(thetaI[Ii+1]))
        tp012[Ii] = np.prod(tp)    
        
        # absorption = e^(-2pi*nd/lamda)

        absor_w[Ii] = np.exp(-2*np.pi*N[Ii].imag*d[Ii]/1044)
        
        print(np.prod(absor_w))
        Ein[Ii+1] = np.array([tp012[Ii]*np.cos(thetaI[Ii+1])*np.sin(PhiIn),
                    ts012[Ii]*np.cos(PhiIn),
                    tp012[Ii]*np.sin(thetaI[Ii+1])*np.sin(PhiIn)])*np.prod(absor_w)
    print(absor_w)
    return OptPar, Kin, Ein 




# 'RefInd_w' : [1, 1.4589+0.00099737j,3.5608+0.00015993j,4.3959+0.1574j],  # air / sio2 / si / Ge
# 'RefInd_2w': [1, 1.4673+0.0020319j, 4.1786+0.038586j,4.753+2.4075j],


