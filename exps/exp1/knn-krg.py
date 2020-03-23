import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
home=os.path.expanduser("~")
sys.path.append(os.path.join(home, 'station2grid'))
from tools import CommonObj, plotMap

from tools.knn_model import KnnModel
from tools.krg_model import KrgModel
from tools.options import *
from tools import datasets, get_predict_result2
##########################################################################################################
##########################################################################################################

info = CommonObj().epa_station_info    
stations = info.SiteEngName[:]
##########################################################################################################
##########################################################################################################
for k in [1,2,3,4,5]:
    for weightKNN in ['uniform','distance']:
        for val_stations in stations:
            print('knn',k,weightKNN,val_stations)
            optionS2GSD = OptionS2GSD(features='pm25', val_stations=val_stations)
            dataS2G = datasets.DataS2G(optionS2GSD)
            dataS2G.setup_test()

            model = KnnModel(dataS2G)
            predict = model.get_values(k, weightKNN).reshape(-1)

            group = 'model_name-knn--val_stations-%s--k-%s--weightKNN-%s'%(val_stations,k,weightKNN)
            result = get_predict_result2(predict, dataS2G, group)

            file_name = '---'.join([group, 'exp1.csv']) ###
            path = os.path.join('results', file_name) 
            result.to_csv(path, index=False)
##########################################################################################################
########################################################################################################## 
for val_stations in stations:
    print('krg',val_stations)
    optionS2GSD = OptionS2GSD(features='pm25', val_stations=val_stations)
    dataS2G = datasets.DataS2G(optionS2GSD)
    dataS2G.setup_test()
    
    model = KrgModel(dataS2G)
    predict = model.get_values().reshape(-1)
    
    group = 'model_name-krg--val_stations-%s'%(val_stations)
    result = get_predict_result2(predict, dataS2G, group)
    
    file_name = '---'.join([group, 'exp1.csv']) ###
    path = os.path.join('results', file_name) 
    result.to_csv(path, index=False)