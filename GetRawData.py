
import numpy as np
import pandas as pd

def GetRawData(gDataOpt):
    
    srng = gDataOpt['smooth_range']
    
    # smooth radius
    srd = int((srng-1)/2)
    
    load_file_name = gDataOpt['load_file_name']
    print(load_file_name)
    #usecols -> read data from 0 to 3 columns (avoid string)
    DataRead = np.loadtxt(load_file_name,usecols=(range(gDataOpt['Total Col']))) 
    x = DataRead[:,gDataOpt['XCol']] *gDataOpt['XAdj']
    yRaw = DataRead[:,gDataOpt['YCol']]
    
    # Shift to that the min(rawdata) is at the 0 deg.
    # yMinInd = np.where(yRaw==min(yRaw))
    # yRaw = np.roll(yRaw,-yMinInd[0]) # [0] in case there are same minimum
    
    # Smooth
    if srng != 1 :
        yExt = np.concatenate((yRaw[-srd:],yRaw,yRaw[0:srd]),axis=None)
        ySmooth = np.convolve(yExt,np.ones(srng)/srng,'valid')
    else:
        ySmooth = yRaw
    
    data = {
        'x': x,
        'yRaw': yRaw,
        'ySmooth':ySmooth 
        }
    
    return data

def SaveFitData(data,gDataOpt):
    data_to_save = np.array([
        data['x'],data['yRaw'],data['ySmooth'],data['ysi']]).T
    df = pd.DataFrame(data_to_save,columns=['x','yRaw','ySmooth','ysi'])
    df['name'] = pd.Series(['phi0', 'au', 'ad', 'ab', 'def_theta', 'def_phi', 'def_ratio', 'Tilt_Direction', 'Tilt_Angle', 'fval'])
    df['solution']=pd.Series(data['solution'])
    df.to_csv(gDataOpt['save_file_name'], header=None, index=None, sep='\t')

def SaveFitData_ads(data,gDataOpt):
    data_to_save = np.array([
        data['x'],data['yRaw'],data['ySmooth'],data['ysi']]).T
    df = pd.DataFrame(data_to_save,columns=['x','yRaw','ySmooth','ysi'])
    df['name'] = pd.Series(['phi0', 'au', 'ad', 'ab', 'def_theta', 'def_phi', 'def_ratio', 'Tilt_Direction', 'Tilt_Angle', \
                            'alpha_ads' ,'TiltDirectionAngle_ads', 'TiltAngle_ads', 'ads_phi0', 'fval'])
    df['solution']=pd.Series(data['solution'])
    df.to_csv(gDataOpt['save_file_name'], header=None, index=None, sep='\t')
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from tkinter import Tk     # from tkinter import Tk for Python 3.x
    from tkinter.filedialog import askopenfilename

    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    # show an "Open" dialog box and return the path to the selected file
    load_file_name = askopenfilename(filetypes=[("txt files","*.txt")])
    
    gDataOpt = {
        # Number of column x/y-value at
        'XCol': 0,
        'YCol': 3,
        'Total Col':4,
        # X-value adjustment
        'XAdj': 1,
        # Range of moving average on y data (0: no smooth)
        'smooth_range': 5,
        # GUI_select or local:
        # selection 1 or run through TXT-files in /data_raw
        'load_file_name': load_file_name}
    data = GetRawData(gDataOpt)
    plt.plot(data['x'],data['yRaw'],'.',
             data['x'],data['ySmooth'],'-'
             )
    plt.title(load_file_name.split('/')[-1])

    # print(x)
