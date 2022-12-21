
import numpy as np
import pygad
import BondsPreset as bp
import SBHM as s
import BondsDeform as bd

struture_type = 'Si111'
StructProp_o = bp.BondsPreset(struture_type)

def fitness_func_set(OptPar,data,solution_1):
    x = data['x']
    # x, raw, smooth
    # desired_output = np.concatenate((data['ySmooth'],data['dySmooth']),axis=0)
#####
    desired_output =data['ySmooth']
    max_d = np.max(desired_output)
#####
    # bond deform axis
    def fitness_func(solution, solution_idx):
        
        StructProp_d = parameter_substitution(solution,solution_1)
        
        ysim = s.SBHM(OptPar,StructProp_o,StructProp_d,x)
        
        # dysim = (ysim[2:]-ysim[0:-2])/2
        # fit_output = np.concatenate((ysim,dysim),axis=0)
#####
        # dysim = (ysim[2:]-ysim[0:-2])/2
        fit_output = ysim
#####   
        fitness = np.sqrt(
                          1/(
            np.mean(np.square((fit_output - desired_output)/max_d))+1e-11)
            )######  +0.000011*(90-solution[4])
        # fitness = (np.sum((fit_output-np.mean(fit_output))*(desired_output-np.mean(desired_output)))\
        #                 /np.sqrt((np.sum((fit_output-np.mean(fit_output))**2))*(np.sum((desired_output-np.mean(desired_output))**2))))**2
        # fitness = 1.0/ (
        #     np.mean(((fit_output - desired_output))**2)+0.000001)  ###### add +0.000001
        
        # fitness = 100-fitness
        
        
        return fitness
    return fitness_func

def func_generation(ga_instance):
    # progress check
    # global fitness_check
    
    # if ga_instance.generations_completed == 1:
    #     fitness_check = 0.00001
    # if (ga_instance.generations_completed % 200000) == 0:
    #     if np.abs((ga_instance.best_solution()[1] - fitness_check)/fitness_check) <= 1e-5 :
    #         return "stop"
    #     else :
    #         fitness_check = ga_instance.best_solution()[1]
    #         print("Fitness  = {fitness}".format(fitness=fitness_check))
        
    # fitting termination 
    # if ga_instance.best_solution()[1] >= 300:
    #     print("Satisfied criteria at step:{}".format(ga_instance.generations_completed))
    #     # print()
    #     return "stop"
    return None
    
def parameter_substitution(solution,solution_1):
    ## Parameters given
    # from SBHMFitPy import BondsPreset
    StructProp_c = bp.BondsPreset(struture_type)
    StructProp_c['phi0'] = solution[0]    # deg.
    StructProp_c['alpha_u'] = solution[1]
    StructProp_c['alpha_d'] = solution[2]
    StructProp_c['alpha_b'] = solution[3]
    # StructProp_c['tau'] = solution[9]######
    # StructProp_c['C'] = solution[10]#####
    defProp = {
        'direction_theta':solution[4],
        'direction_phi':  solution[5],
        'defRatio':       solution[6]}   # percentage
    TiltProp = {
        'TiltDirectionAngle': solution[7],    # deg.
        'TiltAngle': solution[8],
        }    # deg.
    
    StructProp_d = bd.Deformation(defProp,TiltProp, StructProp_c)
    return StructProp_d

def output_from_solution(solution, solution_1,OptPar, data):
    StructProp_d = parameter_substitution(solution,solution_1)
    output = s.SBHM(OptPar,StructProp_o,StructProp_d,data['x'])
    
    return output
        
def pygad2_fit(OptPar, data, solution_1):
    # ysm = data['ySmooth']

    fitness_func = fitness_func_set(OptPar, data, solution_1)

    # symmetry = bp.BondsPreset(struture_type)['symmetry']
    # max alpha (for surface & bulk)
    # a_max = (np.max(ysm))*1   #############np.sqrt
    
    fitness_function = fitness_func

    # Number of solutions (i.e. chromosomes) within the population.
    sol_per_pop  = 20 ###灑幾個點
    # Number of genes in the solution/chromosome.
    num_generations = 20000
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
    # mutation_percent_genes = 30
    mutation_probability = [0.6,0.2]
    # random_mutation_min_val = -90
    # random_mutation_max_val = 90
    
    
    # Par=[phi0,au,ad,ab,defdir_theta,defdir_phi,defRatio, TiltDirectionAngle, TiltAngle]
    initial_population = np.concatenate((
        [np.random.uniform(low=solution_1[0]-1, high=solution_1[0]+1, size=sol_per_pop )],    #phi0
        [np.random.uniform(low=0.9*solution_1[1], high=1.1*solution_1[1], size=sol_per_pop )],  #au
        [np.random.uniform(low=0.9*solution_1[2], high=1.1*solution_1[2], size=sol_per_pop )],  #ad
        [np.random.uniform(low=0.9*solution_1[3], high=1.1*solution_1[3], size=sol_per_pop )],  #ab
        [np.random.uniform(low=0, high=90, size=sol_per_pop )],   #defdir_theta  ##跟defdir_phi 互補
        [np.random.uniform(low=0, high=360, size=sol_per_pop )],   #defdir_phi
        [np.random.uniform(low=0, high=10, size=sol_per_pop )],  #defRatio
        [np.random.uniform(low=solution_1[7]-5, high=solution_1[7]+5, size=sol_per_pop )],   #TiltDirectionAngle
        [np.random.uniform(low=solution_1[8]-0.1, high=solution_1[8]+0.1, size=sol_per_pop )]),   #TiltAngle ##跟TiltDirectionAngle 互補
        # [np.random.uniform(low=0, high=360, size=sol_per_pop)]),   #tau
        # [np.random.uniform(low=-20, high=20, size=sol_per_pop)]),    #C
        axis=0).T
    #####
    # initial_population[0] = [solution_1[0],solution_1[1],solution_1[2],solution_1[3],\
    #                          solution_1[4],solution_1[5],solution_1[6],solution_1[7],\
    #                                          solution_1[8]]
    #####
    gene_space = [
        {'low':solution_1[0]-1,'high':solution_1[0]+1},   # phi0   ##### None
        {'low':0.9*solution_1[1],'high':1.1*solution_1[1]}, # au
        {'low':0.9*solution_1[2],'high':1.1*solution_1[2]}, # ad
        {'low':0.9*solution_1[3],'high':1.1*solution_1[3]}, # ab
        {'low':70,'high':90}, # defdir theta
        {'low':0,'high':360}, # defdir phi
        {'low':0,'high':10}, # defRatio
        {'low':solution_1[7]-5,'high':solution_1[7]+5},  # TiltDirectionAngle
        {'low':solution_1[8]-0.1,'high':solution_1[8]+0.1} # TiltAngle
        # {'low':0,'high':360} ] #tau
        # None]                   #C
        ]
    
    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_function,
        initial_population=initial_population,
        gene_space=gene_space,
        parent_selection_type=parent_selection_type,
        K_tournament = K_tournament,
        keep_parents=keep_parents,
        keep_elitism = keep_elitism,
        # random_mutation_min_val=random_mutation_min_val,
        # random_mutation_max_val = random_mutation_max_val,
        crossover_type=crossover_type,
        crossover_probability=crossover_probability,
        mutation_type=mutation_type,
        mutation_probability = mutation_probability,
        on_generation=func_generation
        # mutation_percent_genes=mutation_percent_genes
        )
        # stop_criteria=[ "saturate_8000"])
 
    ga_instance.run()
    
    ga_instance.plot_fitness()
    
    solution, fval, solution_idx = ga_instance.best_solution()
    
    print("Fitted parameters [au,ab, defRatio]:")
    print("  ".join('{sol:.2f}'.format(sol=k) for k in solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=fval))
    print("Number of generations passed is {generations_completed}".format(generations_completed=ga_instance.generations_completed))
    # print("Initial Population")
    # print(ga_instance.initial_population)
    # print("Final Population")
    # print(ga_instance.population)
    ysi = output_from_solution(solution, solution_1, OptPar, data)
    
    return ysi, fval, solution 

if __name__ == '__main__':
    import OpticParamPreset as opp
    import GetRawData as gd
    import matplotlib.pyplot as plt
    from tkinter import Tk     # from tkinter import Tk for Python 3.x
    from tkinter.filedialog import askopenfilename
    import copy
    import os
    import Fitting_for_3alpha as f
    # import cProfile

    gDataOpt = {
        # % Number of column x/y-value at
        'XCol': 0,
        'YCol': 2,
        # % X-value adjustment
        'XAdj': 1,
        # % Range of moving average on y data (0: no smooth)
        'smooth_range': 1,
        # GUI_select or local:
        # selection 1 or run through TXT-files in /data_raw
        'load_file_name': 'GUI_select'}  
    # Optical Parameter
    
    
    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    # show an "Open" dialog box and return the path to the selected file
    for i in range(1):
        gDataOpt['load_file_name'] = askopenfilename(filetypes=[("txt files","*.txt")])
        gDataOpt['save_file_name'] =\
            os.path.splitext(gDataOpt['load_file_name'])[0]+"_fit.txt"
        gDataOpt['savepar_file_name'] =\
            os.path.splitext(gDataOpt['load_file_name'])[0]+"_fitpar.txt"
            
        ######   
        if np.any(np.array(
                os.path.splitext(gDataOpt['load_file_name'])[0].split('_')) == 'ps'):
            OptPar = opp.OpticParamPreset('ps')
        elif np.any(np.array(
                os.path.splitext(gDataOpt['load_file_name'])[0].split('_')) == 'pp'):
            OptPar = opp.OpticParamPreset('pp')    
        elif np.any(np.array(
                os.path.splitext(gDataOpt['load_file_name'])[0].split('_')) == 'sp'):
            OptPar = opp.OpticParamPreset('sp')
        elif np.any(np.array(
                os.path.splitext(gDataOpt['load_file_name'])[0].split('_')) == 'ss'):
            OptPar = opp.OpticParamPreset('ss')    
        # #########
        
        # x, raw, smooth
        data = gd.GetRawData(gDataOpt)
        data1 = copy.deepcopy(data)
        for i in range(2):
            [data1['ysi'],data1['fval'],data1['solution']] = f.pygad_fit(OptPar, data1)
            
            plt.plot(data1['x'],data1['yRaw'],'.',
                     data1['x'],data1['ySmooth'],'-',
                     data1['x'],data1['ysi'])
            
            [data['ysi'],data['fval'],data['solution']] = pygad2_fit(OptPar, data, data1['solution'])
            # cProfile.run("[data['ysi'],data['fval'],data['solution']] = pygad_fit(OptPar, data)")
        
            plt.plot(data['x'],data['yRaw'],'.',
                     data['x'],data['ySmooth'],'-',
                     data['x'],data['ysi'])
            
            gd.SaveFitData(data,gDataOpt)