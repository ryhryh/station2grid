import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
home=os.path.expanduser("~")
sys.path.append(os.path.join(home, 'station2grid'))

import tools
from tools import options, datasets, knn_model, krg_model

#------------------------------------------------------------------------------
# iterate over all epa stations
#------------------------------------------------------------------------------ 
info = tools.CommonObj().epa_station_info
stations = info.SiteEngName


#------------------------------------------------------------------------------
# KNN
#------------------------------------------------------------------------------  
'''
for k in [1,2,3,4,5]:
    for weightKNN in ['uniform','distance']:
        for val_stations in stations:
            print('knn', k, weightKNN, val_stations)
            optionS2GSD = options.OptionS2GSD(features='pm25', val_stations=val_stations)
            dataS2G = datasets.DataS2G(optionS2GSD)
            dataS2G.setup_test()

            model = knn_model.KnnModel(dataS2G)
            y_hat = model.get_values(k, weightKNN).reshape(-1)         
            y_true = dataS2G.x_raw[:,dataS2G.valid_info.index,0].reshape(-1)

            group = 'model_name-knn--val_stations-%s--k-%s--weightKNN-%s'%(val_stations, k, weightKNN)
            result = tools.get_predict_result(
                y_true, y_hat, group, val_station_names=val_stations.split('_'))

            file_name = ''.join([group, '.csv']) 
            path = os.path.join('../results', file_name) 
            result.to_csv(path, index=False)
'''            
            
#------------------------------------------------------------------------------
# Kriging
#------------------------------------------------------------------------------  
for val_stations in stations:
    print('kriging', val_stations)
    optionS2GSD = options.OptionS2GSD(features='pm25', val_stations=val_stations)
    dataS2G = datasets.DataS2G(optionS2GSD)
    dataS2G.setup_test()    

    model = krg_model.KrgModel(dataS2G)
    y_hat = model.get_values().reshape(-1)

    y_true = dataS2G.x_raw[:,dataS2G.valid_info.index,0].reshape(-1)

    group = 'model_name-krg--val_stations-%s'%(val_stations)
    result = tools.get_predict_result(y_true, y_hat, group, val_station_names=val_stations.split('_'))

    file_name = ''.join([group, '.csv']) 
    path = os.path.join('../results', file_name) 
    result.to_csv(path, index=False)