import argparse
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

def train(config):
    optionS2C = options.OptionS2C(
        domain= config.domain, k= 3, weightKNN= 'distance',
        batch_size=300, 
        epochs=300, ###
        features=config.feature,
        val_stations=config.valid_station,
        dnn_type='a1', 
        ae_type='code_length-4576',
    )

    print('#'*80)
    print(vars(optionS2C))

    modelS2C = station2code_model.ModelS2C(optionS2C)
    modelS2C.train()
    
def test(config):
    optionS2GSD = options.OptionS2GSD(
        domain= config.domain, k= 3, weightKNN= 'distance',
        features=config.feature,
        val_stations=config.valid_station,
        dnn_type='a1', 
        ae_type='code_length-4576',
    )
    
    print('#'*80)
    print(vars(optionS2GSD))
    
    modelS2GSD = station2gridSD_model.ModelS2GSD(optionS2GSD)
    grids = modelS2GSD.test()
    file_name = 'grids_%s.npy'%(config.valid_station)
    np.save(file_name, grids)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()    
    parser.add_argument('--domain', required=False, type=str, default='sat')
    parser.add_argument('--feature', required=False, type=str, default='pm25')
    parser.add_argument('--valid_station', required=False, type=str, default='Tainan')
    parser.add_argument('--isTrain', required=True, type=int, default=1)
    config = parser.parse_args()
    
    if config.isTrain == 1:
        train(config)
    else:
        test(config)
    