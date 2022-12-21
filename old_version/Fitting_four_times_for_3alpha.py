
import numpy as np
import pygad
import BondsPreset as bp
import SBHM as s
import BondsDeform as bd

struture_type = 'Si111'
StructProp_o = bp.BondsPreset(struture_type)

def fitness_func_set(OptPar,data,sol_3_best):
    x = data['x']
    # x, raw, smooth
    # desired_output = np.concatenate((data['ySmooth'],data['dySmooth']),axis=0)
#####
    desired_output =data['ySmooth']
    max_d = np.max(desired_output)
#####
    # bond deform axis
    def fitness_func(solution, solution_idx):
        
        StructProp_d = parameter_substitution(solution,sol_3_best)
        
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
    
def parameter_substitution(solution,sol_3_best):
    ## Parameters given
    # from SBHMFitPy import BondsPreset
    StructProp_c = bp.BondsPreset(struture_type)
    StructProp_c['phi0'] = sol_3_best[0]    # deg.
    StructProp_c['alpha_u'] = solution[0]
    StructProp_c['alpha_d'] = solution[1]
    StructProp_c['alpha_b'] = solution[2]
    # StructProp_c['tau'] = solution[9]######
    # StructProp_c['C'] = solution[10]#####
    defProp = {
        'direction_theta':solution[3],
        'direction_phi':  solution[4],
        'defRatio':       solution[5]}   # percentage
    TiltProp = {
        'TiltDirectionAngle': sol_3_best[7],    # deg.
        'TiltAngle': sol_3_best[8],
        }    # deg.
    
    StructProp_d = bd.Deformation(defProp,TiltProp, StructProp_c)
    return StructProp_d

def output_from_solution(solution, sol_3_best,OptPar, data):
    StructProp_d = parameter_substitution(solution,sol_3_best)
    output = s.SBHM(OptPar,StructProp_o,StructProp_d,data['x'])
    
    return output
        
def pygad4_fit(OptPar, data, sol_3_best):
    # ysm = data['ySmooth']

    fitness_func = fitness_func_set(OptPar, data, sol_3_best)

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
        # [np.random.uniform(low=0.99*sol_3_best[0], high=1.01*sol_3_best[0], size=sol_per_pop )],    #phi0
        [np.random.uniform(low=0.998*sol_3_best[1], high=1.002*sol_3_best[1], size=sol_per_pop )],  #au
        [np.random.uniform(low=0.997*sol_3_best[2], high=1.003*sol_3_best[2], size=sol_per_pop )],  #ad
        [np.random.uniform(low=0.997*sol_3_best[3], high=1.003*sol_3_best[3], size=sol_per_pop )],  #ab
        [np.random.uniform(low=sol_3_best[4]-2, high=sol_3_best[4]+2, size=sol_per_pop )],   #defdir_theta  ##跟defdir_phi 互補
        [np.random.uniform(low=sol_3_best[5]-2, high=1.01*sol_3_best[5]+2, size=sol_per_pop )],   #defdir_phi
        [np.random.uniform(low=sol_3_best[6]-1, high=sol_3_best[6]+1, size=sol_per_pop )]),  #defRatio
        # [np.random.uniform(low=0.99*sol_3_best[7], high=1.01*sol_3_best[7], size=sol_per_pop )],   #TiltDirectionAngle
        # [np.random.uniform(low=0.95*sol_3_best[8], high=1.05*sol_3_best[8], size=sol_per_pop )]),   #TiltAngle ##跟TiltDirectionAngle 互補
        # [np.random.uniform(low=0, high=360, size=sol_per_pop)]),   #tau
        # [np.random.uniform(low=-20, high=20, size=sol_per_pop)]),    #C
        axis=0).T
    #####
    # initial_population[0] = [sol_3_best[0],sol_3_best[1],sol_3_best[2],sol_3_best[3],\
    #                          sol_3_best[4],sol_3_best[5],sol_3_best[6],sol_3_best[7],\
    #                                          sol_3_best[8]]
    #####
    gene_space = [
        # {'low':0.99*sol_3_best[0],'high':1.01*sol_3_best[0]},   # phi0   ##### None
        {'low':0.998*sol_3_best[1],'high':1.002*sol_3_best[1]}, # au
        {'low':0.997*sol_3_best[2],'high':1.003*sol_3_best[2]}, # ad
        {'low':0.997*sol_3_best[3],'high':1.003*sol_3_best[3]}, # ab
        {'low':sol_3_best[4]-2,'high':1.01*sol_3_best[4]+2}, # defdir theta
        {'low':0.99*sol_3_best[5]-2,'high':1.01*sol_3_best[5]+2}, # defdir phi
        {'low':sol_3_best[6]-1,'high':sol_3_best[6]+1} # defRatio
        # {'low':0.99*sol_3_best[7],'high':1.01*sol_3_best[7]},  # TiltDirectionAngle
        # {'low':0.95*sol_3_best[8],'high':1.05*sol_3_best[8]} # TiltAngle
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
    ysi = output_from_solution(solution, sol_3_best, OptPar, data)
    
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
    import Fitting_two_times_for_3alpha as f2t
    import Fitting_three_times_for_3alpha as f3t
    # import cProfile

    gDataOpt = {
        # % Number of column x/y-value at
        'XCol': 0,
        'YCol': 1,
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

    gDataOpt['load_file_name'] = askopenfilename(filetypes=[("txt files","*.txt")])
    gDataOpt['save_file_name'] =\
        os.path.splitext(gDataOpt['load_file_name'])[0]+"_fit.txt"
    gDataOpt['savepar_file_name'] =\
        os.path.splitext(gDataOpt['load_file_name'])[0]+"_fitpar.txt"
        
    ######   
    if os.path.splitext(gDataOpt['load_file_name'])[0].split('_')[-1] == 'ps':
        OptPar = opp.OpticParamPreset('ps')
    elif os.path.splitext(gDataOpt['load_file_name'])[0].split('_')[-1] == 'pp':
        OptPar = opp.OpticParamPreset('pp')    
    elif os.path.splitext(gDataOpt['load_file_name'])[0].split('_')[-1] == 'sp':
        OptPar = opp.OpticParamPreset('sp')
    elif os.path.splitext(gDataOpt['load_file_name'])[0].split('_')[-1] == 'ss':
        OptPar = opp.OpticParamPreset('ss')  
    # #########
    
    # x, raw, smooth
    data4 = gd.GetRawData(gDataOpt)
    data1 = copy.deepcopy(data4)
    data2 = copy.deepcopy(data4)
    data3 = copy.deepcopy(data4)
    ### keep solution 3 and later choose the maximum
    total_run_times = 2
    #times for three times fitting
    luck_times = 10
    #timess for four times fitting
    final_times = 5
    sol_3 = [0]*luck_times
    #choose top x for each run times
    top = 2
    top_list = [0]*(total_run_times*top)
    
    final_solution = [[0]*final_times for x in range(total_run_times)]  #avoid shallow copy
    for j in range(total_run_times):
        for i in range(luck_times):
            while 1 :
                [data1['ysi'],data1['fval'],data1['solution']] = f.pygad_fit(OptPar, data1)
                plt.plot(data1['x'],data1['yRaw'],'.',
                         data1['x'],data1['ySmooth'],'-',
                         data1['x'],data1['ysi'])
                [data2['ysi'],data2['fval'],data2['solution']] = f2t.pygad2_fit(OptPar, data2, data1['solution'])
                plt.plot(data2['x'],data2['yRaw'],'.',
                         data2['x'],data2['ySmooth'],'-',
                         data2['x'],data2['ysi'])
                if data2['fval'] < 8000 :
                    print (' > 8000 ')
                    [data3['ysi'],data3['fval'],data3['solution']] = f3t.pygad3_fit(OptPar, data3, data1['solution'], data2['solution'])
                    # cProfile.run("[data['ysi'],data['fval'],data['solution']] = pygad_fit(OptPar, data)")
                    list_of_solution = [data2['solution'][0],data3['solution'][0],data3['solution'][1],data3['solution'][2],\
                          data3['solution'][3],data3['solution'][4],data3['solution'][5],data2['solution'][7],data2['solution'][8]]
                    # print("Final Fitted parameters [phi0,au,ad,ab,defdir theta, defdir phi, defRatio, TiltDirectionAngle, TiltAngle ]:")
                    # print("  ".join('{sol:.2f}'.format(sol=k) for k in list_of_solution))
                    list_of_solution.append(data3['fval'])
                    sol_3[i] = list_of_solution
                    
                    plt.plot(data3['x'],data3['yRaw'],'.',
                             data3['x'],data3['ySmooth'],'-',
                             data3['x'],data3['ysi'])
                    gd.SaveFitData(data3,gDataOpt)
                    break # 跳出while
                else :
                    print(' < 8000 ')
        # find the index of maximum fval 
        sol_3 = np.array(sol_3) 
        max_id = np.argmax(sol_3[:luck_times,9])
        sol_3_best = sol_3[max_id]
        ### for four times
        for a in range(final_times):  
            [data4['ysi'],data4['fval'],data4['solution']] = pygad4_fit(OptPar, data4, sol_3_best)
            list_of_solution_4 = [sol_3_best[0],data4['solution'][0],data4['solution'][1],data4['solution'][2],\
                  data4['solution'][3],data4['solution'][4],data4['solution'][5],sol_3_best[7],sol_3_best[8],data4['fval']]
            final_solution[j][a] = list_of_solution_4
            print("Final Fitted parameters [phi0,au,ad,ab,defdir theta, defdir phi, defRatio, TiltDirectionAngle, TiltAngle ]:")
            print("  ".join('{sol:.2f}'.format(sol=k) for k in list_of_solution_4))   
        
    final_solution_array = np.array(final_solution)
    # max_id_final = np.unravel_index(np.argmax(final_solution_array[:,:final_times,9]),\
    #                                 [total_run_times,final_times])  
    # final_solution_best = final_solution_array[max_id_final]
    
    #total run times
    for x in range(total_run_times):
        # top 2 
        for y in range(-1,-1-top,-1):
            # the index of the top 2*total run times 
            id_top = np.argsort(final_solution_array[:,:final_times,9])[x][y]
            top_list[total_run_times*x+y] = final_solution_array[x][id_top].tolist()
            
    top_list = np.array(top_list)
        
        
        
        
        