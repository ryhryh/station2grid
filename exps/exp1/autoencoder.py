import os
import sys
home=os.path.expanduser("~")
sys.path.append(os.path.join(home, 'station2grid'))

from keras import backend as K
import gc

from tools.options import *
from models.grid2code_model import ModelG2C
os.environ["CUDA_VISIBLE_DEVICES"]='0'


for domain in ['air']:
    optionG2C = OptionG2C(
        domain= domain, k= 3, weightKNN= 'distance',
        batch_size=300, 
        epochs=100, ###
        ae_type='code_length-4576')
    print(vars(optionG2C))

    modelG2C = ModelG2C(optionG2C)
    modelG2C.train()
    modelG2C.test()

    K.clear_session() 
    gc.collect() 
    del modelG2C