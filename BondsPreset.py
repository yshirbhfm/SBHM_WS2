
import numpy as np
# from sympy import *

def BondsPreset(strucType):
    '''
    tetrahedral: Si(111), ZnO(0002) 1st layer
    '''

    StructProp = {'alpha_u': 0,
                  'alpha_d': 0,
                  'alpha_b': 0,
                  # Fit C_gard as C*exp(tau)
                  # 'tau': 40,
                  # 'C': 2}
                  'C_grad': 0.96 - 1j*1.15}
    if strucType == 'Si111':
        beta = 109.47/180*np.pi
        # bonds [#bond,xyz]
        # bonds = np.array([[0, 0, 1], [np.sin(beta), 0, np.cos(beta)],
        #                   [-1/2*np.sin(beta), np.sqrt(3)/2 *
        #                   np.sin(beta), np.cos(beta)],
        #                   [-1/2*np.sin(beta), -np.sqrt(3)/2*np.sin(beta), np.cos(beta)],                         
        #                   [0,0,-1],[-np.sin(beta),0,-np.cos(beta)],
        #                   [1/2*np.sin(beta),-np.sqrt(3)/2*np.sin(beta),-np.cos(beta)],
        #                   [1/2*np.sin(beta),np.sqrt(3)/2*np.sin(beta),-np.cos(beta)]]) 
        bonds = np.array([[ 0.0469391 ,  0.02559526,  0.99856978],
                           [-0.03666769, -0.92020882, -0.38970656],
                           [ 0.80893775,  0.46741694, -0.35656854],
                           [-0.8151455 ,  0.4291108 , -0.38910375],
                       [ -0.0469391 ,  -0.02559526,  -0.99856978],
                       [0.03666769, 0.92020882, 0.38970656],
                       [ -0.80893775,  -0.46741694, 0.35656854],
                       [0.8151455 ,  -0.4291108 , 0.38910375]]) 
        
        symmetry = 3   
        
    elif strucType == 'Si100':
        a2 = 0.5774
        bonds = np.array([
            [a2, a2, a2], [a2, -a2, a2], [-a2, a2, a2], [-a2, -a2, a2],
            # [a2, a2, -a2], [a2, -a2, -a2], [-a2, a2, -a2], [-a2, -a2, -a2]
            ])
        
        symmetry = 4    
        
    elif strucType == 'Si111+O2':
         # beta = 109.47/180*np.pi
          # bonds [#bond,xyz]
         # bonds = np.array([[0, 0, 1], [np.sin(beta), 0, np.cos(beta)],
         #                    [-1/2*np.sin(beta), np.sqrt(3)/2 *
         #                     np.sin(beta), np.cos(beta)],
         #                    [-1/2*np.sin(beta), -np.sqrt(3)/2*np.sin(beta), np.cos(beta)],                         
         #                     [0,0,-1],[-np.sin(beta),0,-np.cos(beta)],
         #                     [1/2*np.sin(beta),-np.sqrt(3)/2*np.sin(beta),-np.cos(beta)],
         #                     [1/2*np.sin(beta),np.sqrt(3)/2*np.sin(beta),-np.cos(beta)]])
         symmetry = 3
         bonds = np.array([[ 0.0469391 ,  0.02559526,  0.99856978],
                             [-0.03666769, -0.92020882, -0.38970656],
                             [ 0.80893775,  0.46741694, -0.35656854],
                             [-0.8151455 ,  0.4291108 , -0.38910375],
                         [ -0.0469391 ,  -0.02559526,  -0.99856978],
                         [0.03666769, 0.92020882, 0.38970656],
                         [ -0.80893775,  -0.46741694, 0.35656854],
                         [0.8151455 ,  -0.4291108 , 0.38910375]]) 
         
         # doi/10.1063/1.4870629   no rotation
         bonds_adsorption = np.array([[0,0,0],                              #topmost atom
                                           [-0.02127944,  0.31367483,  0.94929199],
                                           [ 0.30595214, -0.20713179,  0.92924147],
                                           [-0.22068781, -0.16397858,  0.96146134],
                                           [-0.01355194, -0.90551664, -0.42409427],
                                           [-0.77281834,  0.46892699, -0.42762049],
                                           [ 0.78807247,  0.43964153, -0.43087946],
                                           [ 0.00953657, -0.00126716,  0.99995372]])   
         #rotaion
         # bonds_adsorption = np.array([[0,0,0],                              #topmost atom
         #                                 [-0.30598396, -0.02974006,  0.95157204],
         #                                 [ 0.2165246 ,  0.29471719,  0.93073029],
         #                                 [ 0.1705918 , -0.23198559,  0.95764353],
         #                                 [ 0.90190874, -0.01412995, -0.43169545],
         #                                 [-0.47688659, -0.76547312, -0.43200705],
         #                                 [-0.43860015,  0.79516978, -0.41873013],
         #                                 [ 0.00953657, -0.00126716,  0.99995372]])  
         
         StructProp['alpha_ads'] = 0
         StructProp['bonds_adsorption'] = bonds_adsorption
         StructProp['bonds_adsorption_phi0'] = 0
    elif strucType == 'WS2':
        # x,y,z,s=symbols('x y z s') # x:0.659 , y=0.38, z=0.649, s=0.761 
        
        bonds = np.array([
            # [-x,y,z],[-x,y,z],[x,y,-z],
            # [0,-s,-z],[x,y,z],[0,-s,z]
            #0.649 for z
            [-0.65859,0.38023 , -0.64937],[ 0.65859,  0.38023 , -0.64937],[ 0   , -0.76047, -0.64937],
            [-0.65859,  0.38023 ,  0.64937],[0.65859, 0.38023 , 0.64937],[ 0  , -0.76047,  0.64937]
            
            # [0.659,  -0.38 , 0.649],[0.659,  -0.38 ,  -0.649],[ -0.659,  -0.38 , 0.649],
            # [ 0   , 0.761, 0.649],[-0.659, -0.38 , -0.649],[ 0   , 0.761,  -0.649]
            ])
        symmetry = 3
    elif strucType == 'WS2+H2O':
        bonds = np.array([
            # [-x,y,z],[-x,y,z],[x,y,-z],
            # [0,-s,-z],[x,y,z],[0,-s,z]
            #0.649 for z
            [-0.65859,0.38023 , -0.64937],[ 0.65859,  0.38023 , -0.64937],[ 0   , -0.76047, -0.64937],
            [-0.65859,  0.38023 ,  0.64937],[0.65859, 0.38023 , 0.64937],[ 0  , -0.76047,  0.64937]
            
            # [0.659,  -0.38 , 0.649],[0.659,  -0.38 ,  -0.649],[ -0.659,  -0.38 , 0.649],
            # [ 0   , 0.761, 0.649],[-0.659, -0.38 , -0.649],[ 0   , 0.761,  -0.649]
            ])
        bonds_adsorption = np.array([
            #water three kinds of bond from 
            #"Mechanism of charge transfer and its impacts on Fermi-level pinning for gas molecules adsorbed on monolayer WS2"
            # 0.97 angstrom bond length for water 
            [-0.3467805645, 0.2002101065,0],[0.3467805645, 0.2002101065,0],[0.0, -0.4004254785,0],
            [-0.3467805645, 0.2002101065,0],[0.3467805645, 0.2002101065,0],[0.0, -0.4004254785,0]
            
            ])
        StructProp['alpha_ads'] = 10
        StructProp['bonds_adsorption'] = bonds_adsorption
        StructProp['bonds_adsorption_phi0'] = 0
        symmetry = 3
        
         
    StructProp['bonds'] = bonds
    StructProp['symmetry'] = symmetry
    return StructProp


if __name__ == '__main__':

    import OpticParamPreset as opp
    import matplotlib.pyplot as plt
    import SBHM as s

    x = np.array(range(360))
    OptPar, Kin, Ein = opp.OpticParamPreset('ss')
    StructProp = BondsPreset('WS2')
    StructProp['phi0'] = 0
    ysi = s.SBHM(OptPar, StructProp, StructProp, x, Kin, Ein)
    plt.plot(x, np.real(ysi))

