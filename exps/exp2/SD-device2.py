import pandas as pd
import numpy as np
from keras import backend as K
import gc
import os
import sys
home=os.path.expanduser("~")
sys.path.append(os.path.join(home, 'station2grid'))

from tools.options import *
from models.station2code_model import ModelS2C
from models.station2gridSD_model import ModelS2GSD
from tools import CommonObj, get_predict_result
info = CommonObj().epa_station_info    
##########################################################################################################
##########################################################################################################


feature_list = [
    'pm25',
    #'pm25_AMBTEMP',
    #'pm25_RH',
    #'pm25_RAINFALL',
    #'pm25_WINDCOS_WINDSIN',
    #'pm25_AMBTEMP_RAINFALL_WINDCOS_WINDSIN',
    #'pm25_AMBTEMP_RH_RAINFALL_WINDCOS_WINDSIN',
    
    #'pm25_PM10',
    #'pm25_NO2',
    #'pm25_SO2',
    #'pm25_O3',
    #'pm25_PM10_NO2_SO2_O3',
    #'pm25_AMBTEMP_RH_RAINFALL_WINDCOS_WINDSIN_PM10_NO2_SO2_O3'
]

dnn_types = ['a1'] # 'a1','a2','a3','b1','b2','b3',
domains = ['air','sat']
epochs=300

os.environ["CUDA_VISIBLE_DEVICES"]='2' 

#stations = info.SiteEngName[4::73//10] 
stations = ['Cailiao','Pingzhen','Tainan','Linyuan',]
##########################################################################################################
##########################################################################################################
   
for dnn_type in dnn_types: 
    for val_stations in stations: 
        for features in feature_list:     
            for domain in domains: 
                ############################################################
                optionS2C = OptionS2C(
                    domain= domain, k= 3, weightKNN= 'distance',
                    batch_size=300, 
                    epochs=epochs, ###
                    features=features,
                    val_stations=val_stations,
                    dnn_type=dnn_type
                )
                print('#'*80)
                print(vars(optionS2C))
                modelS2C = ModelS2C(optionS2C)
                modelS2C.train()

                K.clear_session() 
                del modelS2C
                ############################################################            
                optionS2GSD = OptionS2GSD(
                    domain= domain, k= 3, weightKNN= 'distance',
                    features=features,
                    val_stations=val_stations,
                    dnn_type=dnn_type
                )
                modelS2GSD = ModelS2GSD(optionS2GSD)
                grids = modelS2GSD.test()

                result = get_predict_result(grids, modelS2GSD)
                file_name = '---'.join([modelS2GSD.group, 'exp3.csv']) ###
                path = os.path.join('results', file_name) 
                result.to_csv(path, index=False)

                K.clear_session() 
                gc.collect() 
                del modelS2GSD
                ############################################################

