
import numpy as np
import LinearOpt as lo

def SBHM(OptPar,StructProp_o, StructProp_d,xDeg, Kin, Ein):
    # bonds [Bonds,xyz]
    bonds = StructProp_o['bonds']
    bonds_d =  StructProp_d['bonds']
    
    # x[angle]
    x = (xDeg + StructProp_d['phi0'])/180*np.pi
    
    # Rotation along Z 
    # [xyz,xyz,angle]
    Rz = np.array([[np.cos(x),-np.sin(x),np.zeros_like(x)],
                    [np.sin(x),np.cos(x),np.zeros_like(x)],
                    [np.zeros_like(x),np.zeros_like(x),np.ones_like(x)]])   
    # [angle,xyz,xyz]
    Rz_azi = np.transpose(Rz,(2,1,0))
    
    # Parameters
    noLayers = len(OptPar['RefInd_w']) 
    araBond_s = [0]*noLayers
    araBond_b = [0]*noLayers 
    
    # Transpose Equation 5 of 10.1088/2040-8978/18/3/035501 
    # Azimuthal Rotated alpha-Bonds [angle,bonds,xyz]
    # No deformation
    araBond_s[noLayers-2] = np.matmul(bonds,Rz_azi)         #surface 
    araBond_b[noLayers-2] = np.matmul(bonds,Rz_azi)         #bulk
    # with deformation
    araBond_s[noLayers-1] = np.matmul(bonds_d,Rz_azi)       #surface 
    araBond_b[noLayers-1] = np.matmul(bonds_d,Rz_azi)       #bulk
    
    # For adsorption
    if 'bonds_adsorption' in StructProp_d:    
        bonds_ads = StructProp_d['bonds_adsorption']
        x_ads = (xDeg + StructProp_d['phi0'] + StructProp_d['bonds_adsorption_phi0'])/180*np.pi 
        Rz_ads = np.array([[np.cos(x_ads),-np.sin(x_ads),np.zeros_like(x_ads)],
                        [np.sin(x_ads),np.cos(x_ads),np.zeros_like(x_ads)],
                        [np.zeros_like(x_ads),np.zeros_like(x_ads),np.ones_like(x_ads)]])
        # [angle,xyz,xyz]
        Rz_azi_ads = np.transpose(Rz_ads,(2,1,0))
        araBond_ads = np.matmul(bonds_ads,Rz_azi_ads)      #adsorption
    
    # Parameters
    noXpts = len(x)
    noBonds = bonds.shape[0]
    P_total = [0]*noLayers
    # Field gradient
    Fg = [None]*noLayers
  
    # Calculate SHG from which layer, Ex: for air/SiO2/Si, Start = 2 -> only from Si
    Start = OptPar['Start']

    # Field gradient of Equation 2 of 10.1088/2040-8978/18/3/035501
    for i in range(Start,noLayers):    
        Fg[i] = - 1j*StructProp_o['C_grad']*np.array(Kin[i])  

    # Fit C_gard as C*exp(tau)
    # Fg = (np.cos(StructProp['tau']/180*np.pi)+1j*np.sin(StructProp['tau']/180*np.pi))*StructProp['C']*np.array(Kin[-1])
    
    
    # Equation 1 of 10.1088/2040-8978/18/3/035501
    # polarization (dipole moment) [angle,bonds,xyz]   
    # Ein [# layer, xyz]
        PInterf = (np.dot(araBond_s[i],Ein[i]).reshape(noXpts,noBonds,-1))**2 *araBond_s[i]
        bond_up = PInterf[:,:,2] > 0
        aPInterf = np.copy(PInterf)
        aPInterf[bond_up] = StructProp_d['alpha_u'] * aPInterf[bond_up]
        aPInterf[np.logical_not(bond_up)] =\
            StructProp_d['alpha_d'] * aPInterf[np.logical_not(bond_up)]  
    # Equation 2 of 10.1088/2040-8978/18/3/035501

        PBulk = (np.dot(araBond_b[i],Ein[i]).reshape(noXpts,noBonds,-1))**2 *\
            np.dot(araBond_b[i],Fg[i]).reshape(noXpts,noBonds,-1) *araBond_b[i] ##########################      
        aPBulk = StructProp_d['alpha_b'] * PBulk
        
        P_total[i] = (aPInterf+aPBulk) #aPInterf+aPBulk
    
    # For adsorpption
    # Ein[0] -> adsorption only happened at surface 
    if 'bonds_adsorption' in StructProp_d:    
        
        Padsorption = (np.dot(araBond_ads,Ein[0]).reshape(noXpts,noBonds,-1))**2 *araBond_ads
        print(Ein[0])
        aPadsorption = StructProp_d['alpha_ads'] * Padsorption
        Isum = lo.Fresnel_SH_ads(OptPar,P_total,aPadsorption,Kin)
    else :
        P_total[i] = (aPInterf+aPBulk) #aPInterf+aPBulk
        Isum = lo.Fresnel_SH(OptPar,P_total,Kin)

    return Isum
if __name__ == '__main__':
    
    import OpticParamPreset as opp
    import BondsPreset as bp
    import matplotlib.pyplot as plt

    for i in range(1):
        x = np.array(range(360))
        OptPar, Kin, Ein = opp.OpticParamPreset('pp')
        StructProp = bp.BondsPreset('WS2+H2O')
        StructProp['phi0'] = 0
        ysi  = SBHM(OptPar,StructProp,StructProp,x, Kin, Ein)

        plt.plot(x,np.real(ysi))

        
    
