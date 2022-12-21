
import numpy as np
import pygad
import BondsPreset as bp
import SBHM as s
import BondsDeform as bd

struture_type = 'Si111'
StructProp_o = bp.BondsPreset(struture_type)

def fitness_func_set(OptPar,data,solution_1,solution_2):
    # x, ysmooth
    x = data['x']
    desired_output =data['ySmooth']
    #for normalize fitness
    max_d = np.max(desired_output)

    def fitness_func(solution, solution_idx):
        StructProp_d = parameter_substitution(solution,solution_1,solution_2)        
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
        print('Third')
        fitness_check = 0.00001
    if (ga_instance.generations_completed % 200000) == 0:
        if np.abs((ga_instance.best_solution()[1] - fitness_check)/fitness_check) <= 1e-5 :
            return "stop"
        else :
            fitness_check = ga_instance.best_solution()[1]
            print("Fitness  = {fitness}".format(fitness=fitness_check))
    
def parameter_substitution(solution,solution_1,solution_2):
    ## Parameters given
    # from SBHMFitPy import BondsPreset
    StructProp_s = bp.BondsPreset(struture_type)
    StructProp_s['phi0'] = solution_2[0]
    StructProp_s['alpha_u'] = solution[0]
    StructProp_s['alpha_d'] = solution_2[2]
    StructProp_s['alpha_b'] = solution[1]
    defProp = {
        'direction_theta':solution[2],
        'direction_phi':  solution[3],
        'defRatio':       solution[4]}
    TiltProp = {
        'TiltDirectionAngle': solution[5],    
        'TiltAngle': solution_2[8],
        }   
    StructProp_d = bd.Deformation(defProp,TiltProp, StructProp_s)
    return StructProp_d

def output_from_solution(solution, solution_1, solution_2, OptPar, data):
    StructProp_d = parameter_substitution(solution,solution_1,solution_2)
    output = s.SBHM(OptPar,StructProp_o,StructProp_d,data['x'])
    
    return output
        
def pygad3_fit(OptPar, data, solution_1, solution_2):
    fitness_func = fitness_func_set(OptPar, data, solution_1, solution_2)

    # Number of solutions (i.e. chromosomes) within the population.
    sol_per_pop  = 20 ###灑幾個點
    # Number of genes in the solution/chromosome.
    num_generations = 10000
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
    
    # Par=[phi0,au,ad,ab,defdir_theta,defdir_phi,defRatio, TiltDirectionAngle, TiltAngle]
    initial_population = np.concatenate((
        # [np.random.uniform(low=solution_1[0], high=solution_1[0], size=sol_per_pop )],                 #phi0
        [np.random.uniform(low=0.97*solution_2[1], high=1.03*solution_2[1], size=sol_per_pop )],         #au
        # [np.random.uniform(low=0.9*solution_2[1], high=1.1*solution_2[1], size=sol_per_pop )],         #ad
        [np.random.uniform(low=0.97*solution_2[3], high=1.03*solution_2[3], size=sol_per_pop )],         #ab
        [np.random.uniform(low=90, high=90, size=sol_per_pop )],                                         #defdir_theta
        # [np.random.uniform(low=solution_2[5]-10, high=1.1*solution_2[5]+10, size=sol_per_pop )],         #defdir_phi
        [np.random.uniform(low=0, high=0, size=sol_per_pop )],         #defdir_phi
        [np.random.uniform(low=0, high=0, size=sol_per_pop )],                                           #defRatio
        [np.random.uniform(low=solution_2[7]-5, high=solution_2[7]+5, size=sol_per_pop )]),              #TiltDirectionAngle
        # [np.random.uniform(low=solution_1[8]-0.005, high=solution_1[8], size=sol_per_pop )]),          #TiltAngle 
        axis=0).T

    gene_space = [
        # {'low':solution_1[0],'high':solution_1[0]},               # phi0
        {'low':0.97*solution_2[1],'high':1.03*solution_2[1]},       # au
        # {'low':0.9*solution_2[1],'high':1.1*solution_2[1]},       # ad
        {'low':0.97*solution_2[3],'high':1.03*solution_2[3]},       # ab
        {'low':90,'high':90},                                       # defdir theta
        # {'low':solution_2[5]-10,'high':1.1*solution_2[5]+10},       # defdir phi
        {'low':0,'high':0},       # defdir phi
        {'low':0,'high':0},                                         # defRatio
        {'low':solution_2[7]-5,'high':solution_2[7]+5}              # TiltDirectionAngle
        # {'low':solution_1[8],'high':solution_1[8]}                # TiltAngle
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
        parallel_processing=['thread', 0]
        )
        # stop_criteria=[ "saturate_8000"])
 
    ga_instance.run()
    ga_instance.plot_fitness()
    
    solution, fval, solution_idx = ga_instance.best_solution()
    
    print("Fitted parameters [au,ab, defRatio]:")
    print("  ".join('{sol:.2f}'.format(sol=k) for k in solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=fval))
    print("Number of generations passed is {generations_completed}".format(generations_completed=ga_instance.generations_completed))
    ysi = output_from_solution(solution, solution_1, solution_2, OptPar, data)
    
    return ysi, fval, solution 

if __name__ == '__main__':
    import OpticParamPreset as opp
    import GetRawData as gd
    import matplotlib.pyplot as plt
    from tkinter import Tk     # from tkinter import Tk for Python 3.x
    from tkinter.filedialog import askopenfilename
    import copy
    import os
    import Fitting as f
    import Fitting_two_times as f2t
    # import cProfile

    gDataOpt = {
        # % Number of column x/y-value at
        'XCol': 0,
        'YCol': 3,
        # % X-value adjustment
        'XAdj': 1,
        'Total Col': 4,
        # % Range of moving average on y data (0: no smooth)
        'smooth_range': 1,
        # GUI_select or local:
        # selection 1 or run through TXT-files in /data_raw
        'load_file_name': 'GUI_select'}  
    # Optical Parameter
 
    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    # show an "Open" dialog box and return the path to the selected file
    gDataOpt['load_file_name'] = askopenfilename(filetypes=[("txt files","*.txt")])
    gDataOpt['save_file_name'] =\
        os.path.splitext(gDataOpt['load_file_name'])[0]+"_fit.txt"
        
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
    
    # Prevent shallow copy
    data3 = gd.GetRawData(gDataOpt)
    data1 = copy.deepcopy(data3)
    data2 = copy.deepcopy(data3)
    ### keep solution 3 and later choose the maximum
    run_times = 1
    sol_3 = [0]*run_times
    for i in range(run_times):
        #First
        [data1['ysi'],data1['fval'],data1['solution']] = f.pygad_fit(OptPar, data1)
        #plot
        ax = plt.subplots()[1]
        plt.plot(data1['x'],data1['yRaw'],'.',
                 data1['x'],data1['ySmooth'],'-',
                 data1['x'],data1['ysi'])
        plt.title(gDataOpt['load_file_name'].split('/')[-1]+'_first')
        #show fval in plot
        plt.text(1.01, 0.98, 'fval = '+ str(round(data1['fval'],3)), transform = ax.transAxes)
        
        #Second
        [data2['ysi'],data2['fval'],data2['solution']] = f2t.pygad2_fit(OptPar, data2, data1['solution'])
        #plot
        ax = plt.subplots()[1]
        plt.plot(data2['x'],data2['yRaw'],'.',
                 data2['x'],data2['ySmooth'],'-',
                 data2['x'],data2['ysi'])
        plt.title(gDataOpt['load_file_name'].split('/')[-1]+'_second')
        #show fval in plot
        plt.text(1.01, 0.98, 'fval = '+ str(round(data2['fval'],3)), transform = ax.transAxes)
        
        #Third
        [data3['ysi'],data3['fval'],data3['solution']] = pygad3_fit(OptPar, data3, data1['solution'], data2['solution'])
        #plot
        ax = plt.subplots()[1]
        plt.plot(data3['x'],data3['yRaw'],'.',
                 data3['x'],data3['ySmooth'],'-',
                 data3['x'],data3['ysi'])
        plt.title(gDataOpt['load_file_name'].split('/')[-1]+'_third')
        #show fval in plot
        plt.text(1.01, 0.98, 'fval = '+ str(round(data3['fval'],3)), transform = ax.transAxes)
        
        #Save Final solution
        list_of_solution = [data2['solution'][0],data3['solution'][0],data2['solution'][2],data3['solution'][1],\
              data3['solution'][2],data3['solution'][3],data3['solution'][4],data3['solution'][5],data2['solution'][8]]
        print("Final Fitted parameters [phi0,au,ad,ab,defdir theta, defdir phi, defRatio, TiltDirectionAngle, TiltAngle ]:")
        print("  ".join('{sol:.2f}'.format(sol=k) for k in list_of_solution))
        list_of_solution.append(data3['fval'])
        sol_3[i] = list_of_solution
        data3['solution'] = np.array(list_of_solution)
        gd.SaveFitData(data3,gDataOpt)
        
    # find the index of maximum fval 
    sol_3 = np.array(sol_3) 
    max_id = np.argmax(sol_3[:run_times,9])
    initial_for_4 = sol_3[max_id]
        
        
        
        
        
        
        
        
        
        
        
        
        