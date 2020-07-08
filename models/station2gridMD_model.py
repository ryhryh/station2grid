from models import base_model, networks
from tools import datasets

import pandas as pd
import os
from glob import glob
import numpy as np
home=os.path.expanduser("~")


class ModelS2GMD(base_model.ModelBase):
    def __init__(self, opt):
        self.opt = opt
        self.setup() 
       
    def setup(self): 
        self.data = datasets.DataS2G(self.opt)
        self.compositeNN = networks.CompositeNN(self.opt)
        self.set_group()
        self.set_path()
    
    def train(self): 
        print('training...')
        
        self.data.setup_train() ###
        x_train, y_train = self.data.x_train, self.data.y_train
        x_valid, y_valid = self.data.x_valid, self.data.y_valid
        
        s2gMD_model = self.compositeNN.define_s2gMD(self.data.lats_lons)
        #s2gMD_model.summary() ###
        
        callbacks = self.get_callbacks(min_delta=0.0001, patience=10) ###
        self.history = s2gMD_model.fit(
            x=x_train, 
            y=y_train,
            validation_data=(x_valid, y_valid),
            verbose=1, ###
            batch_size=self.opt.batch_size,
            epochs=self.opt.epochs,
            shuffle=False,
            callbacks=callbacks)
        
        self.save_history(self.history.history)
        print('finish!')
    
    def test(self): 
        print('testing...')
        self.data.setup_test()
        self.stations = self.data.x_norm_select
        
        s2gMD_model = self.compositeNN.get_s2gMD(self.weight_path)
        grids = s2gMD_model.predict(self.stations)
        grids = self.data.denormalize(grids) ###
        print('finish!')
        return grids
