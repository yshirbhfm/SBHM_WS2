# tilt and deformation

import numpy as np
import copy

def Deformation(defProp,TiltProp,StructProp):
# Tilt

    # bonds [#bond,xyz]
    StructProp_d = copy.deepcopy(StructProp)
    bonds = StructProp_d['bonds']
    
    phi_axis = (TiltProp['TiltDirectionAngle']+90) /180*np.pi
    t = TiltProp['TiltAngle'] /180*np.pi
    # rotation axis [xyz]
    u = [np.cos(phi_axis), np.sin(phi_axis), 0]
    
    c1 = 1-np.cos(t)
    # rotation matrix [xyz,xyz]
    RM = np.array([
        [np.cos(t)+u[0]**2*c1, u[0]*u[1]*c1-u[2]*np.sin(t), u[0]*u[2]*c1+u[1]*np.sin(t)],
        [u[1]*u[0]*c1+u[2]*np.sin(t), np.cos(t)+u[1]**2*c1, u[1]*u[2]*c1-u[0]*np.sin(t)],
        [u[2]*u[0]*c1-u[1]*np.sin(t), u[2]*u[1]*c1+u[0]*np.sin(t), np.cos(t)+u[2]**2*c1]])
    
    tBonds = np.matmul(RM,bonds.T)    # [#bond,xyz]
    bonds = tBonds.T
    
    # For adsorption    
    if 'bonds_adsorption' in StructProp_d: 
        bonds_ads = StructProp_d['bonds_adsorption']
        phi_axis = (TiltProp['TiltDirectionAngle_ads']+TiltProp['TiltDirectionAngle']+90) /180*np.pi
        t = (TiltProp['TiltAngle_ads']+TiltProp['TiltAngle']) /180*np.pi
        # rotation axis [xyz]
        u = [np.cos(phi_axis), np.sin(phi_axis), 0]
        
        c1 = 1-np.cos(t)
        # rotation matrix [xyz,xyz]
        RM = np.array([
            [np.cos(t)+u[0]**2*c1, u[0]*u[1]*c1-u[2]*np.sin(t), u[0]*u[2]*c1+u[1]*np.sin(t)],
            [u[1]*u[0]*c1+u[2]*np.sin(t), np.cos(t)+u[1]**2*c1, u[1]*u[2]*c1-u[0]*np.sin(t)],
            [u[2]*u[0]*c1-u[1]*np.sin(t), u[2]*u[1]*c1+u[0]*np.sin(t), np.cos(t)+u[2]**2*c1]])
        
        tbonds_ads = np.matmul(RM,bonds_ads.T)
        bonds_ads = tbonds_ads.T
        StructProp_d['bonds_adsorption'] = bonds_ads
        
# Deformation

    # deform along all directiion
    defax_t = defProp['direction_theta']/180*np.pi  ### with respect to XY plane
    defax_p = defProp['direction_phi']  /180*np.pi  ### with respect to X axis
    defax = np.array(
        [np.cos(defax_t)*np.cos(defax_p),
         np.cos(defax_t)*np.sin(defax_p),
         np.sin(defax_t)]).reshape(1,3)
    # deformed bonds db_j = b_j + ratio*(s_i*b_i)*s_j
    defbond = bonds + np.kron(
        (defProp['defRatio'])/100 * (bonds*defax).sum(axis=1) , defax.T
        ).T
    # normalize bond unit vector
    defbond = defbond / np.sqrt((defbond**2).sum(axis=1).reshape(-1,1))

    StructProp_d['bonds'] = defbond
    
    return StructProp_d

if __name__ == '__main__':
    import OpticParamPreset as opp
    import BondsPreset as bp
    import SBHM as s
    import matplotlib.pyplot as plt
    
    # from SBHMFitPy import BondsPreset
    struture_type = 'WS2+H2O'
    StructProp_o = bp.BondsPreset(struture_type)

    defProp={
        'defRatio': 0,                      # percentage
        'direction_theta' :90,
        'direction_phi' : 0}
    TiltProp={
        'TiltDirectionAngle':-202,          # azimuthal
        'TiltAngle': -5,
        
# For adsorption
        'TiltDirectionAngle_ads':188.26,    # azimuthal for adsorption
        'TiltAngle_ads': 1.41}
    
    x = np.array(range(360))
    OptPar, Kin, Ein = opp.OpticParamPreset('pp')

    StructProp_o['phi0'] = 52.37
    
    StructProp_d = Deformation(defProp, TiltProp, StructProp_o)
    
    ysi = s.SBHM(OptPar,StructProp_o, StructProp_d,x, Kin, Ein)
    plt.plot(x,np.real(ysi))
    # plt.ylim([150,520])
    