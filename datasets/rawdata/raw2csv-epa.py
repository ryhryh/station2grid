import pandas as pd
import os
import numpy as np
from glob import glob
home=os.path.expanduser("~")

def row2block(row, includeFeatures, excludeStations):
    feature=row.iloc[0,2]
    station=row.iloc[0,1]
    #if feature not in includeFeatures: return 
    if station in excludeStations: return 
    
    date=row.iloc[0,0]
    vals=row.iloc[0,3:].values
    
    block=pd.DataFrame()
    block['dt']=pd.date_range(start=date, periods=24,freq='H')
    block['value']=vals
    block['station']=station
    block['feature']=feature
    block=block[['dt','station','feature','value']]
    return block

def process_value(x):
    if x=='NR': return 0
    try: return float(x)
    except: return np.nan
    
def get_oneSite(exl):
    includeFeatures=['PM2.5','PM10','AMB_TEMP','RH']
    excludeStations=['富貴角','馬公','馬祖','金門']
    oneSite=pd.read_excel(exl, encoding='big5')
    
    oneSite_=pd.DataFrame()
    for i in range(len(oneSite))[:]: #####################
        row=oneSite.iloc[i:i+1,:]
        block=row2block(row, includeFeatures, excludeStations)
        oneSite_=oneSite_.append(block)
    return oneSite_
    
def get_oneYear(year):
    exls=glob(os.path.join(home,'station2grid','datasets','rawdata','epa','%s*'%(year-1911),'*','*xls'))
    oneYear=pd.DataFrame()
    for i,exl in enumerate(exls[:]): #####################
        print(i, len(exls)) ###
        oneSite=get_oneSite(exl)
        oneYear=oneYear.append(oneSite)
        oneYear['value']=oneYear.value.apply(process_value)
        oneYear=oneYear.dropna()
    return oneYear


if __name__=='__main__':
    print('processing epa...')
    ##########################################################################################
    dir_path=os.path.join(os.path.expanduser("~"),'station2grid','datasets','csv','csv_files')
    os.makedirs(dir_path,exist_ok=True)
    
    for year in [2017,2018,2019]: #####################
        print(year)
        csv_path=os.path.join(dir_path,'epa%s.csv'%(year))
        oneYear=get_oneYear(year)
        oneYear.to_csv(csv_path,index=False)
    ##########################################################################################
    print('finish epa!')
          