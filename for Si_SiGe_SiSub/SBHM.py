
import numpy as np
import math
import LinearOpt as lo
import scipy.integrate as integrate # scipy提供的數值積分工具

def SBHM(OptPar,StructProp_L1, StructProp_L2,xDeg, Kin, Ein):
    # Parameters
    noLayers = len(OptPar['RefInd_w']) 
    araBond_s = [0]*noLayers
    araBond_b = [0]*noLayers 
    bonds = [None]*noLayers
    # Calculate SHG from which layer, Ex: for air/SiO2/Si, Start = 2 -> only from Si
    Start = OptPar['Start']
    
    # bonds [Bonds,xyz]
    bonds[Start] = StructProp_L1['bonds']        # No deformation   (Si film)
    bonds[Start+1] =  StructProp_L2['bonds']     # With deformation (SiGe film)
    bonds[Start+2] =  StructProp_L1['bonds']     # No deformation   (Si substrate)
    
    # x[angle]
    x = (xDeg + StructProp_L2['phi0'])/180*np.pi
    
    # Rotation along Z 
    # [xyz,xyz,angle]
    Rz = np.array([[np.cos(x),-np.sin(x),np.zeros_like(x)],
                    [np.sin(x),np.cos(x),np.zeros_like(x)],
                    [np.zeros_like(x),np.zeros_like(x),np.ones_like(x)]])   
    # [angle,xyz,xyz]
    Rz_azi = np.transpose(Rz,(2,1,0))
    
    # Transpose Equation 5 of 10.1088/2040-8978/18/3/035501 
    # Azimuthal Rotated alpha-Bonds [angle,bonds,xyz]
    for b in range(Start,noLayers):
        araBond_s[b] = np.matmul(bonds[b],Rz_azi)         #surface
        araBond_b[b] = np.matmul(bonds[b],Rz_azi)         #bulk
    
    # Parameters
    noXpts = len(x)
    noBonds = bonds[Start].shape[0]  #number of bonds 
    P_total = [0]*noLayers
    # Field gradient
    Fg = [None]*noLayers

    # Field gradient of Equation 2 of 10.1088/2040-8978/18/3/035501
    for i in range(Start,noLayers):    
        Fg[i] = - 1j*StructProp_L1['C_grad']*np.array(Kin[i])      
 
    
    # Parameter for absorption
    d = OptPar['thickness']
    absor_2w = [1] * noLayers
    N2 = OptPar['RefInd_2w']
    
   
    
# L1    
    # Equation 1 of 10.1088/2040-8978/18/3/035501
    # polarization (dipole moment) [angle,bonds,xyz]   
    # Ein [# layer, xyz]
    PInterf = (np.dot(araBond_s[Start],Ein[Start]).reshape(noXpts,noBonds,-1))**2 *araBond_s[Start]
    bond_up = PInterf[:,:,2] > 0
    aPInterf = np.copy(PInterf)
    aPInterf[bond_up] = StructProp_L1['alpha_u'] * aPInterf[bond_up]
    aPInterf[np.logical_not(bond_up)] =\
        StructProp_L1['alpha_d'] * aPInterf[np.logical_not(bond_up)]  
        
        
        
    f1 = lambda x : np.exp(-2*np.pi*N2[Start-1].imag*x/522)    # function to be integrated
    area = integrate.quad(f1, 0, 10)    
        
        
    # Equation 2 of 10.1088/2040-8978/18/3/035501
    PBulk = (np.dot(araBond_b[Start],Ein[Start]).reshape(noXpts,noBonds,-1))**2 *\
        np.dot(araBond_b[Start],Fg[Start]).reshape(noXpts,noBonds,-1) *araBond_b[Start] ##########################      
    aPBulk = StructProp_L1['alpha_b'] * PBulk * d[Start]
    
    # absorption = e^(-2pi*nd/lamda)
    
    absor_2w[Start-1] = math.exp(-2*np.pi*N2[Start-1].imag*d[Start-1]/522)
    P_total[Start] = (aPInterf+aPBulk)*np.prod(absor_2w)
    print(absor_2w)
    
    
    
    
# L2   
    # Equation 1 of 10.1088/2040-8978/18/3/035501
    # polarization (dipole moment) [angle,bonds,xyz]   
    # Ein [# layer, xyz]
    PInterf = (np.dot(araBond_s[Start+1],Ein[Start+1]).reshape(noXpts,noBonds,-1))**2 *araBond_s[Start+1]
    bond_up = PInterf[:,:,2] > 0
    aPInterf = np.copy(PInterf)
    aPInterf[bond_up] = StructProp_L2['alpha_u_L2'] * aPInterf[bond_up]
    aPInterf[np.logical_not(bond_up)] =\
        StructProp_L2['alpha_d_L2'] * aPInterf[np.logical_not(bond_up)]  
        
    # Equation 2 of 10.1088/2040-8978/18/3/035501
    PBulk = (np.dot(araBond_b[Start+1],Ein[Start+1]).reshape(noXpts,noBonds,-1))**2 *\
        np.dot(araBond_b[Start+1],Fg[Start+1]).reshape(noXpts,noBonds,-1) *araBond_b[Start+1] ##########################      
    aPBulk = StructProp_L2['alpha_b_L2'] * PBulk * d[Start+2]
    
    # absorption = e^(-2pi*nd/lamda)
    absor_2w[Start] = math.exp(-2*np.pi*N2[Start].imag*d[Start]/522)
    P_total[Start+1] = (aPInterf+aPBulk)*np.prod(absor_2w)
    print(absor_2w)

# L3  
    # Equation 1 of 10.1088/2040-8978/18/3/035501
    # polarization (dipole moment) [angle,bonds,xyz]   
    # Ein [# layer, xyz]
    PInterf = (np.dot(araBond_s[Start+2],Ein[Start+2]).reshape(noXpts,noBonds,-1))**2 *araBond_s[Start+2]
    bond_up = PInterf[:,:,2] > 0
    aPInterf = np.copy(PInterf)
    aPInterf[bond_up] = StructProp_L1['alpha_u'] * aPInterf[bond_up]
    aPInterf[np.logical_not(bond_up)] =\
        StructProp_L1['alpha_d'] * aPInterf[np.logical_not(bond_up)]
        
    # Equation 2 of 10.1088/2040-8978/18/3/035501
    PBulk = (np.dot(araBond_b[Start+2],Ein[Start+2]).reshape(noXpts,noBonds,-1))**2 *\
        np.dot(araBond_b[Start+2],Fg[Start+2]).reshape(noXpts,noBonds,-1) *araBond_b[Start+2] ##########################      
    aPBulk = StructProp_L1['alpha_b'] * PBulk * d[Start+2]
    
    # absorption = e^(-2pi*nd/lamda)
    absor_2w[Start+1] = math.exp(-2*np.pi*N2[Start+1].imag*d[Start+1]/522)
    P_total[Start+2] = (aPInterf+aPBulk)*np.prod(absor_2w)    
    print(absor_2w)
    
    Isum = lo.Fresnel_SH(OptPar,P_total,Kin)

    return Isum
if __name__ == '__main__':
    
    import OpticParamPreset as opp
    import BondsPreset as bp
    import matplotlib.pyplot as plt

    for i in range(1):
        x = np.array(range(360))
        OptPar, Kin, Ein = opp.OpticParamPreset('pp')
        StructProp = bp.BondsPreset('Si111')
        StructProp['phi0'] = 0
        ysi = SBHM(OptPar,StructProp,StructProp,x, Kin, Ein)

        plt.plot(x,np.real(ysi))

        
    
