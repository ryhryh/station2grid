
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
from joblib import Parallel, delayed

def get_one_dt_epa(dt):
    i_dt=dts.index(dt)
    one_dt=pd.DataFrame()
    for i_s,s in enumerate(info.SiteEngName):
        for i_f,f in enumerate(epa_features):
            dummy=pd.DataFrame()
            dummy=pd.DataFrame()
            dummy['station']=[s]
            dummy['feature']=[f]
            dummy['dt']=[dt]
            dummy['value']=[epa_arr[i_dt,i_s,i_f]]
            one_dt=one_dt.append(dummy)
    return one_dt


info=CommonObj().epa_station_info

path='/media/disk3/feynman52/station2grid/datasets/npy/epa/epa2019.npy'
epa_arr=np.load(path)
epa_arr.shape

path='/media/disk3/feynman52/station2grid/datasets/info/epa-features.csv'
epa_features=pd.read_csv(path).feature.values
epa_features

print('process')
####################################################################################
dts=pd.date_range(start='2019', end='2020', freq='H', closed='left').tolist()
df_dts=Parallel(n_jobs=30)(delayed(get_one_dt_epa)(dt) for dt in dts[:]) ###
df_epa=pd.concat(df_dts,axis=0)

path='df_epa.csv'
df_epa.to_csv(path,index=False)
####################################################################################
print('finish')