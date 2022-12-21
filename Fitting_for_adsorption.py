
import numpy as np
import pygad
import BondsPreset as bp
import SBHM as s
import BondsDeform as bd

struture_type = 'Si111+O2'
StructProp_o = bp.BondsPreset(struture_type)

#from_no_adsorption
phi0 = 82.40    
ad = 0       
ab = 140.09    
def_theta = 90
def_phi = 0
def_ratio = 0
tilt_dir = 138.60   
tilt_angle = 0.84 

def fitness_func_set(OptPar,data):
    # x, ysmooth
    x = data['x']
    desired_output =data['ySmooth']
    #for normalize fitness
    max_d = np.max(desired_output)

    def fitness_func(solution, solution_idx):
        StructProp_d = parameter_substitution(solution)     
        fit_output = s.SBHM(OptPar,StructProp_o,StructProp_d,x)

        fitness = np.sqrt(
                          1/(
            np.mean(np.square((fit_output - desired_output)/max_d))+1e-11)
            )
        return fitness
    return fitness_func

def func_generation(ga_instance):
    # progress check
    global fitness_check
    
    if ga_instance.generations_completed == 1:
        print('First')
        fitness_check = 0.00001
    if (ga_instance.generations_completed % 200000) == 0:
        if np.abs((ga_instance.best_solution()[1] - fitness_check)/fitness_check) <= 1e-5 :
            return "stop"
        else :
            fitness_check = ga_instance.best_solution()[1]
            print("Fitness  = {fitness}".format(fitness=fitness_check))       
    
def parameter_substitution(solution):
    ## Parameters given
    # from SBHMFitPy import BondsPreset
    StructProp_s = bp.BondsPreset(struture_type)
    
    StructProp_s['phi0'] = phi0   
    StructProp_s['alpha_u'] = solution[0]
    StructProp_s['alpha_d'] = ad
    StructProp_s['alpha_b'] = ab
    StructProp_s['alpha_ads'] = solution[1]        # for adsorption
    
    defProp = {
        'direction_theta':def_theta,
        'direction_phi':  def_phi,
        'defRatio':       def_ratio}   
    TiltProp = {
        'TiltDirectionAngle': tilt_dir,   
        'TiltAngle': tilt_angle,
        'TiltDirectionAngle_ads':solution[2],       # for adsorption
        'TiltAngle_ads': solution[3]               # for adsorption
        }

    StructProp_s['bonds_adsorption_phi0'] = solution[4]      # for adsorption 2
    
    StructProp_d = bd.Deformation(defProp,TiltProp, StructProp_s)
    return StructProp_d

def output_from_solution(solution, OptPar, data):
    StructProp_d = parameter_substitution(solution)
    output = s.SBHM(OptPar,StructProp_o,StructProp_d,data['x'])
    
    return output
        
def pygad_fit(OptPar, data):
    ysm = data['ySmooth']
    # symmetry = bp.BondsPreset(struture_type)['symmetry']
    # max alpha (for surface & bulk)
    a_max = (np.max(ysm))*1
    
    fitness_func = fitness_func_set(OptPar, data)

    # Number of solutions (i.e. chromosomes) within the population.
    sol_per_pop  = 20 ###灑幾個點
    # Number of genes in the solution/chromosome.
    num_generations = 20
    parent_selection_type = "tournament"
    K_tournament = 3
    #Number of solutions to be selected as parents.
    num_parents_mating = int(50 /100*sol_per_pop )   ### 50/100
    # Number of parents to keep in the current population.
    # -1: keep all
    keep_elitism  = 0
    keep_parents = int(20 /100*num_parents_mating ) ####30
    crossover_type = "uniform"
    crossover_probability = 0.6
    mutation_type = "adaptive"
    mutation_probability = [0.6,0.2]

    # Par=[au, alpha_ads ,TiltDirectionAngle_ads, TiltAngle_ads, ads_phi0  ]
    initial_population = np.concatenate((
        [np.random.uniform(low=0, high=a_max, size=sol_per_pop )],              #au
        [np.random.uniform(low=-a_max/10, high=a_max/10, size=sol_per_pop )],   # alpha_ads
        [np.random.uniform(low=0, high=0, size=sol_per_pop )],                  #TiltDirectionAngle for adsorption
        [np.random.uniform(low=0, high=0, size=sol_per_pop )],                  #TiltAngle for adsorption
        [np.random.uniform(low=0, high=0, size=sol_per_pop )]),                 # ads_phi0
        axis=0).T
    
    gene_space = [

        {'low':0,'high':a_max},            # au
        {'low':-a_max/10,'high':a_max/10}, # alpha_ads
        {'low':0,'high':0},                # TiltDirectionAngle for adsorption
        {'low':0,'high':0},                # TiltAngle for adsorption
        {'low':0,'high':0}                 # ads_phi0
        ]  
    
    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_func,
        initial_population=initial_population,
        gene_space=gene_space,
        parent_selection_type=parent_selection_type,
        K_tournament = K_tournament,
        keep_parents=keep_parents,
        keep_elitism = keep_elitism,
        crossover_type=crossover_type,
        crossover_probability=crossover_probability,
        mutation_type=mutation_type,
        mutation_probability = mutation_probability,
        on_generation=func_generation,
        parallel_processing = ['thread',0]
        )
        # stop_criteria=[ "saturate_8000"])

    ga_instance.run()
    ga_instance.plot_fitness()
    
    solution, fval, solution_idx = ga_instance.best_solution()

    print("Fitted parameters [au, alpha_ads ,TiltDirectionAngle_ads, TiltAngle_ads, ads_phi0  ]:")
    print("  ".join('{sol:.2f}'.format(sol=k) for k in solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=fval))
    print("Number of generations passed is {generations_completed}".format(generations_completed=ga_instance.generations_completed))
    ysi = output_from_solution(solution, OptPar, data)
    
    return ysi, fval, solution 

if __name__ == '__main__':
    import OpticParamPreset as opp
    import GetRawData as gd
    import matplotlib.pyplot as plt
    from tkinter import Tk     # from tkinter import Tk for Python 3.x
    from tkinter.filedialog import askopenfilename
    import os
    # import cProfile
    import time
    t1 = time.time()
    gDataOpt = {
        # % Number of column x/y-value at
        'XCol': 0,
        'YCol': 2,
        # % X-value adjustment
        'XAdj': 1,
        'Total Col': 3,
        # % Range of moving average on y data (0: no smooth)
        'smooth_range': 1,
        # GUI_select or local:
        # selection 1 or run through TXT-files in /data_raw
        'load_file_name': 'GUI_select'}  
    
    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    # show an "Open" dialog box and return the path to the selected file
    gDataOpt['load_file_name'] = askopenfilename(filetypes=[("txt files","*.txt")])
    gDataOpt['save_file_name'] =\
        os.path.splitext(gDataOpt['load_file_name'])[0]+"_ads_fit.txt"
        
     #Get Polarization
    if np.any(np.array(
            os.path.splitext(gDataOpt['load_file_name'])[0].split('_')) == 'ss'):
        OptPar = opp.OpticParamPreset('ss')
    elif np.any(np.array(
            os.path.splitext(gDataOpt['load_file_name'])[0].split('_')) == 'sp'):
        OptPar = opp.OpticParamPreset('sp')    
    elif np.any(np.array(
            os.path.splitext(gDataOpt['load_file_name'])[0].split('_')) == 'ps'):
        OptPar = opp.OpticParamPreset('ps')
    elif np.any(np.array(
            os.path.splitext(gDataOpt['load_file_name'])[0].split('_')) == 'pp'):
        OptPar = opp.OpticParamPreset('pp') 
    else:
        OptPar = opp.OpticParamPreset('pp')

    data = gd.GetRawData(gDataOpt)
    for i in range(1):

        [data['ysi'],data['fval'],data['solution']] = pygad_fit(OptPar, data)
        # cProfile.run("[data['ysi'],data['fval'],data['solution']] = pygad_fit(OptPar, data)")
        data['solution'] = np.array([phi0,data['solution'][0],ad,ab,def_theta,def_phi,def_ratio,tilt_dir,tilt_angle,\
                                     data['solution'][1],data['solution'][2],data['solution'][3],data['solution'][4],data['fval']])
        ax = plt.subplots()[1]
        plt.plot(data['x'],data['yRaw'],'.',
                 data['x'],data['ySmooth'],'-',
                 data['x'],data['ysi'])
        plt.title(gDataOpt['load_file_name'].split('/')[-1])
        #show fval in plot
        plt.text(1.01, 0.98, 'fval = '+ str(round(data['fval'],3)), transform = ax.transAxes)
        plt.show()
        gd.SaveFitData_ads(data,gDataOpt)
        
    t2 = time.time()
    print("Time is", t2-t1)
    