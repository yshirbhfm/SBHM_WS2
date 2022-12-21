
import numpy as np


def BondsPreset(strucType):
    '''
    tetrahedral: Si(111), ZnO(0002) 1st layer
    '''

    StructProp = {'alpha_u': 10,
                  'alpha_d': 0,
                  'alpha_b': 0.005,
                  # Fit C_gard as C*exp(tau)
                  # 'tau': 40,
                  # 'C': 2}
                  'C_grad': 0.96 - 1j*1.15}
    if strucType == 'Si111':
        beta = 109.47/180*np.pi
        # bonds [#bond,xyz]
        bonds = np.array([[0,0,1], [np.sin(beta), 0, np.cos(beta)],
                          [-1/2*np.sin(beta), np.sqrt(3)/2 *
                          np.sin(beta), np.cos(beta)],
                          [-1/2*np.sin(beta), -np.sqrt(3)/2*np.sin(beta), np.cos(beta)],                         
                          [0,0,-1],[-np.sin(beta),0,-np.cos(beta)],
                          [1/2*np.sin(beta),-np.sqrt(3)/2*np.sin(beta),-np.cos(beta)],
                          [1/2*np.sin(beta),np.sqrt(3)/2*np.sin(beta),-np.cos(beta)]]) 

        symmetry = 3   
        
    elif strucType == 'Si100':
        a2 = 0.5774
        bonds = np.array([
            [a2, a2, a2], [a2, -a2, a2], [-a2, a2, a2], [-a2, -a2, a2],
            [a2, a2, -a2], [a2, -a2, -a2], [-a2, a2, -a2], [-a2, -a2, -a2]])
        
        symmetry = 4    
         
    StructProp['bonds'] = bonds
    StructProp['symmetry'] = symmetry
    StructProp['alpha_u_L2'] = 10
    StructProp['alpha_d_L2'] = 0
    StructProp['alpha_b_L2'] = 0.001
    return StructProp


if __name__ == '__main__':

    import OpticParamPreset as opp
    import matplotlib.pyplot as plt
    import SBHM as s

    x = np.array(range(360))
    OptPar, Kin, Ein = opp.OpticParamPreset('pp')
    StructProp = BondsPreset('Si111')
    StructProp['phi0'] = 0
    ysi = s.SBHM(OptPar, StructProp, StructProp, x, Kin, Ein)
    plt.plot(x, np.real(ysi))

