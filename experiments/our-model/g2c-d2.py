import os
import sys
home=os.path.expanduser("~")
sys.path.append(os.path.join(home, 'station2grid'))

from tools import options
from models import grid2code_model

from keras import backend as K
import gc

os.environ["CUDA_VISIBLE_DEVICES"]='2'


#------------------------------------------------------------------------------
# train grid2code
#------------------------------------------------------------------------------ 
for domain in ['sat']:
    for i in [6864]: # 2288, 3432, 5720, 6864
        ae_type='code_length-%s'%(i)
        optionG2C = options.OptionG2C(
            domain= domain, k= 3, weightKNN= 'distance',
            batch_size=300, 
            epochs=200, ###
            ae_type=ae_type)
        print(vars(optionG2C))
        modelG2C = grid2code_model.ModelG2C(optionG2C)
        
        
        modelG2C.train()
        modelG2C.test()

        K.clear_session() 
        gc.collect() 
        del modelG2C