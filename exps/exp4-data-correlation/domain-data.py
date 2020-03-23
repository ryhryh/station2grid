
# coding: utf-8

# In[1]:


from glob import glob
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
matplotlib.rcParams['axes.unicode_minus']=False

import os
import sys
home=os.path.expanduser("~")
sys.path.append(os.path.join(home, 'station2grid'))

from tools import plotMap, CommonObj
import re


info=CommonObj().epa_station_info


domain='air'


domain_features=pd.read_csv('/media/disk3/feynman52/station2grid/datasets/info/%s-features.csv'%(domain),index_col=0).feature.values


path='/media/disk3/feynman52/station2grid/datasets/npy/%s/domain_%s-k_3-weightKNN_distance/station'%(domain,domain)
station_paths=sorted(glob(os.path.join(path,'*')))



def get_one_dt(station_path):
    pat='station/(.+)_'
    tar=station_path
    dt=pd.to_datetime(re.search(pat,tar).group(1))
    station=np.load(station_path)
    
    if dt.minute==30: return # remove 10:30 because epa w/o half hour
    include_cols=['pm25', 'pm10', 'temperature', 'humidity', 'PM25', 'PM10',
        'AMBTEMP', 'RH']
    
    one_dt=pd.DataFrame()
    for i_s,s in enumerate(info.SiteEngName):
        for i_f,f in enumerate(domain_features):
            #if f not in include_cols: continue ###
            dummy=pd.DataFrame()
            dummy['station']=[s]
            dummy['feature']=[f]
            dummy['dt']=[dt]
            dummy['value']=[station[0,i_s,i_f]]
            one_dt=one_dt.append(dummy)
    return one_dt 


import time
from math import sqrt
from joblib import Parallel, delayed

t=time.time()
print('process')

dts=Parallel(n_jobs=20)(delayed(get_one_dt)(station_path) for station_path in station_paths[:])
all_dt=pd.concat(dts,axis=0)

print('finish')
print(time.time()-t)



path='df_domain.csv'
all_dt.to_csv(path,index=False)
