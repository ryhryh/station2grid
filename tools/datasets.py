from glob import glob
from random import randint
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod
import os
home = os.path.expanduser("~")
#import sys
#sys.path.append(os.path.join(home, 'station2grid'))

from tools import CommonObj
##########################################################################################
##########################################################################################
class DataBase(ABC):
    def __init__(self, opt):
        self.opt = opt
        self.setup()
        
    @abstractmethod
    def setup(self): pass
    
    def set_base_dir(self):
        source = 'domain_%s-k_%s-weightKNN_%s'%(self.opt.domain, self.opt.k, self.opt.weightKNN)
        self.base_dir = os.path.join(home, 'station2grid', 'datasets', 'npy', self.opt.domain, source)       
    
    def set_generator(self):
        self.x_train_paths, self.x_valid_paths, self.y_train_paths, self.y_valid_paths = \
        train_test_split(self.x_paths, self.y_paths ,test_size=0.2, random_state=42)     
        
        self.g_train = self.get_generator(self.x_train_paths, self.y_train_paths)
        self.g_valid = self.get_generator(self.x_valid_paths, self.y_valid_paths) 
    
    def get_generator(self, x_paths, y_paths):
        while True:
            batch_x = []
            batch_y = []
            for i in range(self.opt.batch_size):
                j=randint(0, len(x_paths)-1)
                
                x_path = x_paths[j] 
                x_arr = self.x_path2arr(x_path)
                batch_x.append(x_arr)
                
                y_path = y_paths[j] 
                y_arr = self.y_path2arr(y_path)
                batch_y.append(y_arr)
                
            batch_x = np.concatenate(batch_x, axis=0)
            batch_y = np.concatenate(batch_y, axis=0)
            yield batch_x, batch_y
            
    def grid_path2arr(self, path):
        arr = np.load(path)
        arr = arr[...,:1]
        arr = self.normalize(arr) 
        return arr
            
    def station_path2arr(self, path):
        arr = np.load(path)
        arr = arr[:, self.i_stations, :][..., self.i_features]
        #arr = self.normalize(arr) ###normalize
        return arr
    
    def code_path2arr(self, path):
        arr = np.load(path)
        arr = arr.reshape(1, -1)
        return arr
    
    def normalize(self, arr):
        norm_const = 100
        arr = arr / norm_const
        arr = arr.astype('float')
        return arr
    
    def denormalize(self, arr):
        norm_const = 100
        arr = norm_const * arr
        arr = arr.astype('float')
        return arr
    
    def setup_i_stations(self):        
        self.info = CommonObj().epa_station_info
        val_stations = self.opt.val_stations.split('_') 
        c = self.info.SiteEngName.isin(val_stations)
        self.train_info = self.info[~c]
        self.valid_info = self.info[c]
        
        self.i_stations = self.train_info.index.tolist()
        self.lats_lons = self.train_info[['row','col']].values 
        
    def setup_i_features(self):
        path = os.path.join(home,'station2grid','datasets','info','%s-features'%(self.opt.domain))
        domain_features = pd.read_csv(path).feature.tolist()
        features = self.opt.features.split('_')
        self.i_features = [domain_features.index(feature) for feature in features]
        #print(domain_features, features, self.i_features)

##########################################################################################
##########################################################################################        
class DataG2C(DataBase):
    def __init__(self, opt):
        super().__init__(opt)
        
    def setup(self):
        self.set_base_dir()
        self.set_paths()
        self.set_generator()
    
    def set_paths(self):
        self.x_paths = sorted(glob(os.path.join(self.base_dir, 'grid', '*')))[:] ###
        self.x_path2arr = self.grid_path2arr
        self.y_paths = sorted(glob(os.path.join(self.base_dir, 'grid', '*')))[:] ###
        self.y_path2arr = self.grid_path2arr
        
##########################################################################################
##########################################################################################
class DataS2C(DataBase):
    def __init__(self, opt):
        super().__init__(opt)
        
    def setup(self):
        self.setup_i_stations()
        self.setup_i_features()
        
        self.set_base_dir()
        self.set_paths()
        self.set_generator()
    
    def set_paths(self):
        self.x_paths = sorted(glob(os.path.join(self.base_dir, 'station', '*minmax_norm.npy'))) ###
        self.x_path2arr = self.station_path2arr
        self.y_paths = sorted(glob(os.path.join(self.base_dir, 'code', self.opt.ae_type, '*'))) 
        self.y_path2arr = self.code_path2arr
               
##########################################################################################
##########################################################################################            
class DataS2G(DataBase):
    def __init__(self, opt):
        super().__init__(opt)
        
    def setup(self):
        self.setup_i_stations()
        self.setup_i_features()
    
    def setup_i_features(self):
        path = os.path.join(home,'station2grid','datasets','info','epa-features')
        domain_features = pd.read_csv(path).feature.tolist()
        
        domain2epa = {
            'pm25':'PM25',
            'pm10':'PM10',
            'temperature':'AMBTEMP',
            #'humidity':'RH'
        }
        features=self.opt.features
        for old in domain2epa: features = features.replace(old, domain2epa[old])
        features = features.split('_')
        
        self.i_features = [domain_features.index(feature) for feature in features]
    
    
    def setup_train(self):
        path = os.path.join(home,'station2grid','datasets','npy','epa','train_epa_addmou.npy') 
        self.x_raw = np.load(path) 
        
        path = os.path.join(home,'station2grid','datasets','npy','epa',
                            'train_epa_addmou_minmax_norm.npy') 
        self.x_raw_norm = np.load(path) ###
        
        self.x_all = self.x_raw_norm[:, self.i_stations, :][:, :, self.i_features] 
        self.y_all = self.x_raw_norm[:, self.i_stations, 0].reshape(-1, len(self.i_stations))
        
        self.x_train, self.x_valid, self.y_train, self.y_valid = \
        train_test_split(self.x_all, self.y_all, test_size=0.2, random_state=42)

    def setup_test(self): 
        path = os.path.join(home,'station2grid','datasets','npy','epa','test_epa_addmou.npy') 
        self.x_raw = np.load(path)
        self.x_raw_select = self.x_raw[:,self.i_stations,:][:,:,self.i_features] # x_raw_select
                
        path = os.path.join(home,'station2grid','datasets','npy','epa','test_epa_addmou_minmax_norm.npy') 
        self.x_norm = np.load(path)
        self.x_norm_select = self.x_norm[:,self.i_stations,:][:,:,self.i_features]   # x_norm_select    

        