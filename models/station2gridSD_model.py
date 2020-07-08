from models import base_model, grid2code_model, station2code_model 
from tools import datasets, options
import os
home=os.path.expanduser("~")


class ModelS2GSD(base_model.ModelBase):
    def __init__(self, opt):
        self.opt = opt
        self.setup()    
        
    def setup(self):
        self.data = datasets.DataS2G(self.opt)
        self.set_group()
        
        optionS2C = options.OptionS2C(domain=self.opt.domain, 
                                      k=self.opt.k, 
                                      weightKNN=self.opt.weightKNN, 
                                      features=self.opt.features, 
                                      val_stations=self.opt.val_stations, 
                                      ae_type=self.opt.ae_type, 
                                      dnn_type=self.opt.dnn_type)   
        
        modelS2C = station2code_model.ModelS2C(optionS2C)###
        self.s2c_model = modelS2C.dnn.get_dnn(modelS2C.weight_path)
        
        optionG2C = options.OptionG2C(domain=self.opt.domain, 
                                      k=self.opt.k, 
                                      weightKNN=self.opt.weightKNN, 
                                      ae_type=self.opt.ae_type, )  
        modelG2C = grid2code_model.ModelG2C(optionG2C)###
        self.c2g_model = modelG2C.autoencoder.get_decoder(modelG2C.weight_path)
        
    def test(self): 
        print('testing')
        self.data.setup_test()
        self.stations = self.data.x_norm_select
        
        codes = self.s2c_model.predict(self.stations)
        
        code_channel_length = int(self.opt.ae_type[len('code_length-'):len('code_length-')+1])
        print('code_channel_length',code_channel_length)###
        codes = codes.reshape(-1, 44, 26, code_channel_length)
        
        grids_norm = self.c2g_model.predict(codes)
        grids_denorm = self.data.denormalize(grids_norm)
        
        print('finish')
        return grids_denorm
