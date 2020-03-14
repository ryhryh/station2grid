from models.networks import AE, FCNN
from tools.data_generator import DataGenerator
import pandas as pd
import os
from glob import glob
import numpy as np
home=os.path.expanduser("~")

class Station2GridSD():
    def __init__(self, opt):
        self.opt = opt
        self.ae = AE(opt)
        self.fcnn = FCNN(opt)
        self.dataGenerator = DataGenerator(opt)
        
        self.n_val_stations = len(self.opt.val_stations.split('_')) ###
        self.n_features = len(self.opt.features.split('_')) ###
        self.setup_weight_dir()
    
    def test(self): 
        print('testing...')
        dataGenerator = self.dataGenerator
        
        g2c_weights = sorted(glob(os.path.join(self.g2c_weight_dir, '*hdf5')))
        g2c_weight = g2c_weights[-1]
        s2c_weights = sorted(glob(os.path.join(self.s2c_weight_dir, '*hdf5')))
        s2c_weight = s2c_weights[-1]

        c2g_model = self.ae.get_decoder(g2c_weight)
        s2c_model = self.fcnn.get_fcnn(s2c_weight)
        
        self.stations = dataGenerator.station_path2arr(self.opt.epa_station_path)
        
        codes=s2c_model.predict(self.stations)
        codes=codes.reshape(-1, 44, 26, 4)
        
        grids=c2g_model.predict(codes)
        grids=dataGenerator.denormalize(grids)
        print('finish!')
        return grids
    
    def save_history(self, history):
        df_history = pd.DataFrame(history.history)
        path = os.path.join(self.weight_dir, 'history.csv',)
        df_history.to_csv(path, index=False)
        
    def setup_weight_dir(self):
        opt = self.opt
        source = 'domain_%s-k_%s-weightKNN_%s'%(opt.domain, opt.k, opt.weightKNN)
        base_dir = os.path.join(home, 'station2grid', 'weights', 'single', source)
        self.g2c_weight_dir = os.path.join(base_dir, 'grid2code', opt.ae_type)
        self.s2c_weight_dir = os.path.join(base_dir, 'station2code', opt.ae_type, str(self.n_val_stations), opt.val_stations, opt.features)
        
        