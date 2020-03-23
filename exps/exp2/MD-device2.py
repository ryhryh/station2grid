import pandas as pd
import numpy as np
from keras import backend as K
import gc
import os
import sys
home=os.path.expanduser("~")
sys.path.append(os.path.join(home, 'station2grid'))

from tools.options import *
from models.station2gridMD_model import ModelS2GMD
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
    #'pm25_AMBTEMP_RH_RAINFALL_WINDCOS_WINDSIN',
    
    #'pm25_PM10',
    #'pm25_NO2',
    #'pm25_SO2',
    #'pm25_O3',
    #'pm25_PM10_NO2_SO2_O3',
    #'pm25_AMBTEMP_RH_RAINFALL_WINDCOS_WINDSIN_PM10_NO2_SO2_O3'
]

dnn_types = ['a1'] # 'a1','a2','a3','b1','b2','b3',
#domains = ['air','sat']
epochs=300

os.environ["CUDA_VISIBLE_DEVICES"]='2' 

stations = ['Cailiao','Pingzhen','Tainan','Linyuan',]

composite_types = ['c0','c1','c2','c3'] #'c0','c1','c2','c3'


##########################################################################################################


for composite_type in composite_types: # 4
    for val_stations in stations[:]:  ### 4 
        for features in feature_list[:]:   ###   1  
            ############################################################
            optionS2GMD = OptionS2GMD(
                features=features,
                val_stations=val_stations,
                batch_size=300, epochs=50, ###
                ae_type='code_length-4576', dnn_type='a1',
                domains='air_3_distance~sat_3_distance', 
                composite_type=composite_type, ###
            ) 

            print('#'*80)
            print(vars(optionS2GMD))
            modelS2GMD = ModelS2GMD(optionS2GMD)
            modelS2GMD.train()

            K.clear_session() 
            ############################################################            
            grids = modelS2GMD.test()

            result = get_predict_result(grids, modelS2GMD)
            file_name = '---'.join([modelS2GMD.group, 'exp3.csv']) ###
            path = os.path.join('results', file_name) 
            result.to_csv(path, index=False)

            K.clear_session() 
            gc.collect() 
            del modelS2GMD
            ############################################################

