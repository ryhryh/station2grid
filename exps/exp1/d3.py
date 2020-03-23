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
from models.station2gridMD_model import ModelS2GMD

from tools import CommonObj, get_predict_result
os.environ["CUDA_VISIBLE_DEVICES"]='3' 
########################################################################################################
########################################################################################################


info = CommonObj().epa_station_info    

feature_list = [
    'pm25',
    #'pm25_PM10_NO2_SO2_O3',
    #'pm25_AMBTEMP_RH_RAINFALL_WINDCOS_WINDSIN',
    #'pm25_AMBTEMP_RH_RAINFALL_WINDCOS_WINDSIN_PM10_NO2_SO2_O3',
]


domain_list = ['sat']

#station_list = info.SiteEngName.values[0::3]



station_list = [
    'Changhua'
    #'Dabajianshan','Mushan','Yushan','Beidawushan','Guting','Banqiao','Zhongli',
    #'Xitun','Tainan','Fengshan','Yilan','Hualien','Taitung','Hengchun'
] 

########################################################################################################

for repeat in range(4,6):
    for val_stations in station_list: 
        for features in feature_list:  
            for domain in domain_list: 

                ############################################################  

                optionS2C = OptionS2C(
                    domain= domain, k= 3, weightKNN= 'distance',
                    batch_size=300, 
                    epochs=200, ###
                    features=features,
                    val_stations=val_stations,
                    dnn_type='a1'
                )
                print('#'*80)
                print(vars(optionS2C))
                modelS2C = ModelS2C(optionS2C)
                modelS2C.train()

                K.clear_session() 
                del modelS2C

                ############################################################  
                model_name = 'minmax_repeat%s'%(repeat)

                optionS2GSD = OptionS2GSD(
                    model_name = model_name,
                    domain= domain, k= 3, weightKNN= 'distance',
                    features=features,
                    val_stations=val_stations,
                    dnn_type='a1'
                )
                modelS2GSD = ModelS2GSD(optionS2GSD)
                grids = modelS2GSD.test()

                ############################################################
                valid_info = modelS2GSD.data.valid_info
                valid_idxs = valid_info.index.values
                val_rows = valid_info.row.values
                val_cols = valid_info.col.values

                y_true = modelS2GSD.data.x_raw[:,valid_idxs,0]
                y_hat = grids[:,val_rows,val_cols,0]
                group = modelS2GSD.group
                val_station_names = val_stations.split('_')

                result = get_predict_result(y_true, y_hat, group, val_station_names=val_station_names)

                file_name = '---'.join([modelS2GSD.group, '.csv']) ###
                path = os.path.join('results', file_name) 
                result.to_csv(path, index=False)

                ############################################################
                K.clear_session() 
                gc.collect() 
                del modelS2GSD



########################################################################################################
'''
composite_type_list = ['c0'] #'c0','c1','c2','c3'

for composite_type in composite_type_list: # 4
    for val_stations in station_list: # 13
        for features in feature_list: # 2   
            ############################################################
            optionS2GMD = OptionS2GMD(
                model_name = 's2gmd_minmax',
                features=features,
                val_stations=val_stations,
                batch_size=100, epochs=60, ### 
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
            
            ############################################################
            valid_info = modelS2GMD.data.valid_info
            valid_idxs = valid_info.index.values
            val_rows = valid_info.row.values
            val_cols = valid_info.col.values
            
            y_true = modelS2GMD.data.x_raw[:,valid_idxs,0]
            y_hat = grids[:,val_rows,val_cols,0]
            group = modelS2GMD.group
            val_station_names = val_stations.split('_')
            
            result = get_predict_result(y_true, y_hat, group, val_station_names=val_station_names)
            
            file_name = '---'.join([modelS2GMD.group, '.csv']) ###
            path = os.path.join('results', file_name) 
            result.to_csv(path, index=False)

            ############################################################

            K.clear_session() 
            gc.collect() 
            del modelS2GMD
            
'''