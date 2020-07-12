import argparse
import os
import sys
home=os.path.expanduser("~")
sys.path.append(os.path.join(home, 'station2grid'))

from tools import options
from models import grid2code_model

from keras import backend as K
import gc

os.environ["CUDA_VISIBLE_DEVICES"]='0'


def main(config):
    optionG2C = options.OptionG2C(
        domain= config.domain, k= 3, weightKNN= 'distance',
        batch_size=300, 
        epochs=200, ###
        ae_type='code_length-4576')
    print(vars(optionG2C))
    modelG2C = grid2code_model.ModelG2C(optionG2C)

    modelG2C.train()
    modelG2C.test()

    K.clear_session() 
    gc.collect() 
    del modelG2C
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()  
    parser.add_argument('--domain', required=False, type=str, default='sat')
    config = parser.parse_args()
    
    main(config)
