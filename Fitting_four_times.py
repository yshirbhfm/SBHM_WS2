
import numpy as np
import pygad
import BondsPreset as bp
import SBHM as s
import BondsDeform as bd
import pandas as pd 

struture_type = 'Si111'
StructProp_o = bp.BondsPreset(struture_type)

def fitness_func_set(OptPar,data,sol_3_best):
    # x, ysmooth
    x = data['x']
    desired_output =data['ySmooth']
    #for normalize fitness
    max_d = np.max(desired_output)
    
    def fitness_func(solution, solution_idx):        
        StructProp_d = parameter_substitution(solution,sol_3_best)
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
        print('Fourth')
        fitness_check = 0.00001
    if (ga_instance.generations_completed % 200000) == 0:
        if np.abs((ga_instance.best_solution()[1] - fitness_check)/fitness_check) <= 1e-5 :
            return "stop"
        else :
            fitness_check = ga_instance.best_solution()[1]
            print("Fitness  = {fitness}".format(fitness=fitness_check))
    
def parameter_substitution(solution,sol_3_best):
    ## Parameters given
    # from SBHMFitPy import BondsPreset
    StructProp_s = bp.BondsPreset(struture_type)
    StructProp_s['phi0'] = sol_3_best[0]    
    StructProp_s['alpha_u'] = solution[0]
    StructProp_s['alpha_d'] = sol_3_best[2]
    StructProp_s['alpha_b'] = solution[1]
    defProp = {
        'direction_theta':solution[2],
        'direction_phi':  solution[3],
        'defRatio':       solution[4]}
    TiltProp = {
        'TiltDirectionAngle': sol_3_best[7],   
        'TiltAngle': sol_3_best[8],
        }    
    StructProp_d = bd.Deformation(defProp,TiltProp, StructProp_s)
    return StructProp_d

def output_from_solution(solution, sol_3_best,OptPar, data):
    StructProp_d = parameter_substitution(solution,sol_3_best)
    output = s.SBHM(OptPar,StructProp_o,StructProp_d,data['x'])
    
    return output
        
def pygad4_fit(OptPar, data, sol_3_best):
    fitness_func = fitness_func_set(OptPar, data, sol_3_best)

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
    mutation_probability = [0.6,0.2]
    
    # Par=[phi0,au,ad,ab,defdir_theta,defdir_phi,defRatio, TiltDirectionAngle, TiltAngle]
    initial_population = np.concatenate((
        # [np.random.uniform(low=0.99*sol_3_best[0], high=1.01*sol_3_best[0], size=sol_per_pop )],               #phi0
        [np.random.uniform(low=0.998*sol_3_best[1], high=1.002*sol_3_best[1], size=sol_per_pop )],               #au
        # [np.random.uniform(low=0, high=0, size=sol_per_pop )],                                                 #ad
        [np.random.uniform(low=0.997*sol_3_best[3], high=1.003*sol_3_best[3], size=sol_per_pop )],               #ab
        # [np.random.uniform(low=sol_3_best[4]-5, high=sol_3_best[4]+5, size=sol_per_pop )],                       #defdir_theta
        # [np.random.uniform(low=sol_3_best[5]-5, high=sol_3_best[5]+5, size=sol_per_pop )],                       #defdir_phi
        # [np.random.uniform(low=sol_3_best[6]-1, high=sol_3_best[6]+1, size=sol_per_pop )]),                      #defRatio
        [np.random.uniform(low=90, high=90, size=sol_per_pop )],                       #defdir_theta
        [np.random.uniform(low=0, high=0, size=sol_per_pop )],                       #defdir_phi
        [np.random.uniform(low=0, high=0, size=sol_per_pop )]),                      #defRatio
        # [np.random.uniform(low=0.99*sol_3_best[7], high=1.01*sol_3_best[7], size=sol_per_pop )],               #TiltDirectionAngle
        # [np.random.uniform(low=0.95*sol_3_best[8], high=1.05*sol_3_best[8], size=sol_per_pop )]),              #TiltAngle 
        axis=0).T
    
    gene_space = [
        # {'low':0.99*sol_3_best[0],'high':1.01*sol_3_best[0]},         # phi0   
        {'low':0.998*sol_3_best[1],'high':1.002*sol_3_best[1]},         # au
        # {'low':0,'high':0},                                           # ad
        {'low':0.997*sol_3_best[3],'high':1.003*sol_3_best[3]},         # ab
        # {'low':sol_3_best[4]-5,'high':sol_3_best[4]+5},                 # defdir theta
        # {'low':sol_3_best[5]-5,'high':sol_3_best[5]+5},                 # defdir phi
        # {'low':sol_3_best[6]-1,'high':sol_3_best[6]+1}                  # defRatio
        {'low':90,'high':90},                 # defdir theta
        {'low':0,'high':0},                 # defdir phi
        {'low':0,'high':0}                  # defRatio
        # {'low':0.99*sol_3_best[7],'high':1.01*sol_3_best[7]},         # TiltDirectionAngle
        # {'low':0.95*sol_3_best[8],'high':1.05*sol_3_best[8]}          # TiltAngle
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
    import pandas as pd
    import time
    import Fitting as f
    import Fitting_two_times as f2t
    import Fitting_three_times as f3t
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

    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    # show an "Open" dialog box and return the path to the selected file

    gDataOpt['load_file_name'] = askopenfilename(filetypes=[("txt files","*.txt")])
         
    t1 = time.time()

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
    
    #fval choose to pass
    fval_limit = 100
    # Prevent shallow copy
    data4 = gd.GetRawData(gDataOpt)
    data1 = copy.deepcopy(data4)
    data2 = copy.deepcopy(data4)
    data3 = copy.deepcopy(data4)
    data_save = copy.deepcopy(data4)
    
    # keep solution 3 and later choose the maximum
    #Total times
    total_run_times = 2
    #times for three times fitting
    luck_times = 3
    #timess for four times fitting
    final_times = 3
    sol_3 = [0]*luck_times
    #choose top x for each run times
    top = 2
    top_list = [0]*(total_run_times*top) 
    final_solution = [[0]*final_times for x in range(total_run_times)]  #avoid shallow copy  (for solution)
    # final_data = [[0]*final_times for x in range(total_run_times)]      #avoid shallow copy  (for data save and plot)
    df = pd.DataFrame(data4['x'],columns=['x'])
    df['yRaw'] = data4['yRaw']
    df['ySmooth'] = data4['ySmooth']
    for TRT in range(total_run_times):
        for i in range(luck_times):
            while 1 :
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
                if data1['fval'] > fval_limit :
                    print(' > ' +str(fval_limit))
                    
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
                    [data3['ysi'],data3['fval'],data3['solution']] = f3t.pygad3_fit(OptPar, data3, data1['solution'], data2['solution'])

                    #plot
                    ax = plt.subplots()[1]
                    plt.plot(data3['x'],data3['yRaw'],'.',
                             data3['x'],data3['ySmooth'],'-',
                             data3['x'],data3['ysi'])
                    plt.title(gDataOpt['load_file_name'].split('/')[-1]+'_third')
                    #show fval in plot
                    plt.text(1.01, 0.98, 'fval = '+ str(round(data3['fval'],3)), transform = ax.transAxes)
                    list_of_solution = [data2['solution'][0],data3['solution'][0],data2['solution'][2],data3['solution'][1],\
                          data3['solution'][2],data3['solution'][3],data3['solution'][4],data3['solution'][5],data2['solution'][8]]
                    list_of_solution.append(data3['fval'])
                    sol_3[i] = list_of_solution
                    break
                else :
                    print(' < '+str(fval_limit))
        # find the index of maximum fval 
        sol_3 = np.array(sol_3) 
        max_id = np.argmax(sol_3[:luck_times,9])
        sol_3_best = sol_3[max_id]
        
        #Fourth
        for FT in range(final_times):  
            [data4['ysi'],data4['fval'],data4['solution']] = pygad4_fit(OptPar, data4, sol_3_best)
            #plot
            ax = plt.subplots()[1]
            plt.plot(data4['x'],data4['yRaw'],'.',
                     data4['x'],data4['ySmooth'],'-',
                     data4['x'],data4['ysi'])
            plt.title(gDataOpt['load_file_name'].split('/')[-1]+'_Fourth')
            #show fval in plot
            plt.text(1.01, 0.98, 'fval = '+ str(round(data4['fval'],3)), transform = ax.transAxes)
            
            #Save final solution
            list_of_solution_4 = [sol_3_best[0],data4['solution'][0],sol_3_best[2],data4['solution'][1],\
                  data4['solution'][2],data4['solution'][3],data4['solution'][4],sol_3_best[7],sol_3_best[8],data4['fval']]
            final_solution[TRT][FT] = list_of_solution_4
            df['ysi_'+str(final_times*TRT+FT)] = data4['ysi']
            print("Final Fitted parameters [phi0,au,ad,ab,defdir theta, defdir phi, defRatio, TiltDirectionAngle, TiltAngle ]:")
            print("  ".join('{sol:.2f}'.format(sol=k) for k in list_of_solution_4))         
            
    final_solution_array = np.array(final_solution)
    
    #total run times
    for x in range(total_run_times):
        # top 2 
        for y in range(-1,-1-top,-1):
            # the index of the top 2*total run times 
            id_top = np.argsort(final_solution_array[:,:final_times,9])[x][y]
            top_list[top*x+y] = final_solution_array[x][id_top].tolist()
            
            #save data
            gDataOpt['save_file_name'] =\
                os.path.splitext(gDataOpt['load_file_name'])[0]+"_fit_"+str(x)+'_'+str(abs(y))+".txt"
            data_save['ysi'] = df['ysi_'+str(final_times*x+id_top)]
            data_save['solution'] = top_list[top*x+y]
            gd.SaveFitData(data_save,gDataOpt)
            
    top_list = np.array(top_list)
    top_df = pd.DataFrame(top_list,columns=['phi0', 'au', 'ad', 'ab', 'def_theta', 'def_phi', 'def_ratio', 'Tilt_Direction', 'Tilt_Angle', 'fval'])
    
    t2 = time.time()
    print("Time is", t2-t1)
        
        
        