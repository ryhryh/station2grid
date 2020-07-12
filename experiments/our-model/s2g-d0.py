import pandas as pd
import numpy as np
from keras import backend as K
import gc
import os
import sys
home=os.path.expanduser("~")
sys.path.append(os.path.join(home, 'station2grid'))

import tools
from tools import options
from models import station2code_model, station2gridSD_model, station2gridMD_model

os.environ["CUDA_VISIBLE_DEVICES"]='0' 

#-----------------------------------------------------------------------------
# iterate over all epa stations
#-----------------------------------------------------------------------------
feature_list = [
    #'pm25', 
    'pm25_AMBTEMP_RH_RAINFALL_WINDCOS_WINDSIN_PM10_NO2_SO2_O3'
]

info = tools.CommonObj().epa_station_info
#station_list = info.SiteEngName
station_list = ['Tainan', 'Guting', 'Xitun', 'Zuoying', 'Hualien']


#-----------------------------------------------------------------------------
# S2GSD
#-----------------------------------------------------------------------------
domain_list = ['sat']

for i in [2288, 3432, 4576, 5720, 6864]:
    for val_stations in station_list:  
        for features in feature_list:  
            for domain in domain_list: 
                
                ae_type='code_length-%s'%(i)

                #---------------------------------------
                # train station2code
                #---------------------------------------
                optionS2C = options.OptionS2C(
                    domain= domain, k= 3, weightKNN= 'distance',
                    batch_size=300, 
                    epochs=300, ###
                    features=features,
                    val_stations=val_stations,
                    dnn_type='a1', 
                    ae_type=ae_type,
                )

                print('#'*80)
                print(vars(optionS2C))

                modelS2C = station2code_model.ModelS2C(optionS2C)
                modelS2C.train()

                K.clear_session() 
                del modelS2C

                #---------------------------------------
                # infer station2gridSD
                #---------------------------------------  
                optionS2GSD = options.OptionS2GSD(
                    domain= domain, k= 3, weightKNN= 'distance',
                    features=features,
                    val_stations=val_stations,
                    dnn_type='a1',
                    ae_type=ae_type,
                )
                modelS2GSD = station2gridSD_model.ModelS2GSD(optionS2GSD)
                grids = modelS2GSD.test()

                #---------------------------------------
                # pick valid station from grids_hat 
                #---------------------------------------
                valid_info = modelS2GSD.data.valid_info
                valid_idxs = valid_info.index.values
                val_rows = valid_info.row.values
                val_cols = valid_info.col.values

                y_true = modelS2GSD.data.x_raw[:,valid_idxs,0]
                y_hat = grids[:,val_rows,val_cols,0]
                group = modelS2GSD.group
                val_station_names = val_stations.split('_')

                result = tools.get_predict_result(
                    y_true, y_hat, group, val_station_names=val_station_names)

                file_name = '---'.join([modelS2GSD.group, '.csv']) ###
                path = os.path.join('../results', file_name) 
                result.to_csv(path, index=False)

                ############################################################
                K.clear_session() 
                gc.collect() 
                del modelS2GSD
            
'''
#-----------------------------------------------------------------------------
# S2GMD
#-----------------------------------------------------------------------------
composite_type_list = ['c0'] 

for composite_type in composite_type_list: 
    for val_stations in station_list: 
        for features in feature_list: 
            
            #---------------------------------------
            # train station2gridMD
            #---------------------------------------  
            optionS2GMD = options.OptionS2GMD(
                features=features,
                val_stations=val_stations,
                batch_size=100, epochs=100,  
                ae_type='code_length-4576', dnn_type='a1',
                domains='air_3_distance~sat_3_distance', 
                composite_type=composite_type, 
            ) 

            print('#'*80)
            print(vars(optionS2GMD))
            
            modelS2GMD = station2gridMD_model.ModelS2GMD(optionS2GMD)
            modelS2GMD.train()

            K.clear_session() 
            
            #---------------------------------------
            # infer station2gridMD
            #---------------------------------------              
            grids = modelS2GMD.test()
            
            
            #---------------------------------------
            # pick valid station from grids_hat 
            #---------------------------------------
            valid_info = modelS2GMD.data.valid_info
            valid_idxs = valid_info.index.values
            val_rows = valid_info.row.values
            val_cols = valid_info.col.values
            
            y_true = modelS2GMD.data.x_raw[:,valid_idxs,0]
            y_hat = grids[:,val_rows,val_cols,0]
            group = modelS2GMD.group
            val_station_names = val_stations.split('_')
            
            result = tools.get_predict_result(y_true, y_hat, group, val_station_names=val_station_names)
            
            file_name = ''.join([modelS2GMD.group, '.csv']) ###
            path = os.path.join('../results', file_name) 
            result.to_csv(path, index=False)

            ############################################################

            K.clear_session() 
            gc.collect() 
            del modelS2GMD
            '''