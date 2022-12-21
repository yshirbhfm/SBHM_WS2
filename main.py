
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory
import OpticParamPreset as opp
import GetRawData as gd
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import copy
import Fitting as f
import Fitting_two_times as f2t
import Fitting_three_times as f3t
import Fitting_four_times as f4t


def fit_one_time(gDataOpt, OptPar) :
    data = gd.GetRawData(gDataOpt)
    for i in range(2):
        [data['ysi'],data['fval'],data['solution']] = f.pygad_fit(OptPar, data)
        # cProfile.run("[data['ysi'],data['fval'],data['solution']] = pygad_fit(OptPar, data)")
        
        ax = plt.subplots()[1]
        plt.plot(data['x'],data['yRaw'],'.',
                 data['x'],data['ySmooth'],'-',
                 data['x'],data['ysi'])
        plt.title(gDataOpt['load_file_name'].split('/')[-1])
        #show fval in plot
        plt.text(1.01, 0.98, 'fval = '+ str(round(data['fval'],3)), transform = ax.transAxes)
        gDataOpt['save_file_name'] =\
            os.path.sep.join([gDataOpt['save_dir'],os.path.basename(gDataOpt['load_file_name']).split('.')[0]+"_SBHM fit_.txt"])
        gd.SaveFitData(data,gDataOpt)
        
def fit_two_times(gDataOpt, OptPar) :
    # Prevent shallow copy
    data = gd.GetRawData(gDataOpt)
    data1 = copy.deepcopy(data)
    for i in range(1):
        #First
        [data1['ysi'],data1['fval'],data1['solution']] = f.pygad_fit(OptPar, data1)
        
        ax = plt.subplots()[1]
        plt.plot(data1['x'],data1['yRaw'],'.',
                 data1['x'],data1['ySmooth'],'-',
                 data1['x'],data1['ysi'])
        plt.title(gDataOpt['load_file_name'].split('/')[-1]+'_first')
        #show fval in plot
        plt.text(1.01, 0.98, 'fval = '+ str(round(data1['fval'],3)), transform = ax.transAxes)
        
        #Second
        [data['ysi'],data['fval'],data['solution']] = f2t.pygad2_fit(OptPar, data, data1['solution'])
        # cProfile.run("[data['ysi'],data['fval'],data['solution']] = pygad_fit(OptPar, data)")
        ax = plt.subplots()[1]
        plt.plot(data['x'],data['yRaw'],'.',
                 data['x'],data['ySmooth'],'-',
                 data['x'],data['ysi'])
        plt.title(gDataOpt['load_file_name'].split('/')[-1]+'_second')
        #show fval in plot
        plt.text(1.01, 0.98, 'fval = '+ str(round(data['fval'],3)), transform = ax.transAxes)
        gDataOpt['save_file_name'] =\
            os.path.sep.join([gDataOpt['save_dir'],os.path.basename(gDataOpt['load_file_name']).split('.')[0]+"_SBHM fit_.txt"])
        gd.SaveFitData(data,gDataOpt)
    
def fit_three_times(gDataOpt, OptPar) :
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
        [data3['ysi'],data3['fval'],data3['solution']] = f3t.pygad3_fit(OptPar, data3, data1['solution'], data2['solution'])
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
        gDataOpt['save_file_name'] =\
            os.path.sep.join([gDataOpt['save_dir'],os.path.basename(gDataOpt['load_file_name']).split('.')[0]+"_SBHM fit_.txt"])
        gd.SaveFitData(data3,gDataOpt)
        

def fit_four_times(gDataOpt, OptPar) :
    t1 = time.time()
    
    #fval choose to pass
    fval_limit = 30
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
    luck_times = 5
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
            [data4['ysi'],data4['fval'],data4['solution']] = f4t.pygad4_fit(OptPar, data4, sol_3_best)
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
                os.path.sep.join([gDataOpt['save_dir'],os.path.basename(gDataOpt['load_file_name']).split('.')[0]+"_SBHM fit_"+str(x)+'_'+str(abs(y))+".txt"])
            data_save['ysi'] = df['ysi_'+str(final_times*x+id_top)]
            data_save['solution'] = top_list[top*x+y]
            gd.SaveFitData(data_save,gDataOpt)
            
    top_list = np.array(top_list)
    top_df = pd.DataFrame(top_list,columns=['phi0', 'au', 'ad', 'ab', 'def_theta', 'def_phi', 'def_ratio', 'Tilt_Direction', 'Tilt_Angle', 'fval'])       
    t2 = time.time()
    print("Time is", t2-t1)



gDataOpt = {
    # % Number of column x/y-value at
    'XCol': 0,
    'YCol': 3,
    # % X-value adjustment
    'XAdj': 1.125,
    'Total Col': 4,
    # % Range of moving average on y data (0: no smooth)
    'smooth_range': 1,
    # 'GUI_select' or 'local':
    # selection 1 or run through TXT-files in /data_raw
    'load_file_name': 'GUI_select_folder'}   
    
gDataOpt_2 = copy.deepcopy(gDataOpt)

if gDataOpt['load_file_name']=='local':
    data_dir = os.path.join(os.path.sep,"sspp111")
    save_dir = os.path.join(os.path.sep,"fit_result")
    all_data = os.listdir(os.getcwd()+data_dir)
    for data_file in all_data:
        if data_file.endswith(".txt"):  # Ensures reading only JPG files.
            gDataOpt['load_file_name'] = os.path.sep.join([os.getcwd(),data_dir,data_file])
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
                
            fit_four_times(gDataOpt, OptPar)
            
elif gDataOpt['load_file_name']=='GUI_select_folder':
    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    folder_selected = askdirectory()
    gDataOpt['save_dir'] = os.path.sep.join(
        [folder_selected,"fit_raw data_result"])
    if not os.path.exists(gDataOpt['save_dir']):
        os.makedirs(gDataOpt['save_dir'])
    all_data = os.listdir(folder_selected)
    for data_file in all_data:
        if data_file.endswith(".txt"):  # Ensures reading only JPG files.
            gDataOpt['load_file_name'] = os.path.sep.join([folder_selected,data_file])
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
            fit_four_times(gDataOpt, OptPar)    
    
elif gDataOpt['load_file_name']=='GUI_select':
    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    # show an "Open" dialog box and return the path to the selected file
    gDataOpt['load_file_name'] = askopenfilename(filetypes=[("txt files","*.txt")])

    fit_four_times(gDataOpt, OptPar)

