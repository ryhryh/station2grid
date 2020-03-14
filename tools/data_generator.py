from glob import glob
import os
from random import randint
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
home = os.path.expanduser("~")

class DataGenerator():
    def __init__(self,opt):
        self.opt=opt
        self.model_name = opt.model_name
        self.batch_size = opt.batch_size
        self.setup()
        self.generator_train = self.get_generator(self.x_train_paths, self.y_train_paths)
        self.generator_valid = self.get_generator(self.x_valid_paths, self.y_valid_paths) 
    
    def setup(self):
        self.setup_base_dir()
        
        if self.model_name == 'grid2code':
            self.x_paths = self.get_paths('grid')
            self.x_path2arr = self.grid_path2arr
            self.y_paths = self.get_paths('grid')
            self.y_path2arr = self.grid_path2arr
        elif self.model_name == 'station2code':
            self.setup_i_stations() ###
            self.setup_i_features(self.opt.domain) ###
            self.x_paths = self.get_paths('station')
            self.x_path2arr = self.station_path2arr
            self.y_paths = self.get_paths('code')
            self.y_path2arr = self.code_path2arr
        elif self.model_name == 'station2gridSD':
            self.setup_i_stations() ###
            self.setup_i_features('epa') ###
            self.x_paths = self.get_paths('station')
            self.x_path2arr = self.station_path2arr
            self.y_paths = self.get_paths('code')
            self.y_path2arr = self.code_path2arr
            
        self.x_train_paths, self.x_valid_paths, self.y_train_paths, self.y_valid_paths = train_test_split(self.x_paths, self.y_paths ,test_size=0.2, random_state=42)
        
            
    def get_generator(self, x_paths, y_paths):
        x_path2arr, y_path2arr = self.x_path2arr, self.y_path2arr
        batch_size = self.batch_size
        while True:
            batch_x = []
            batch_y = []
            for i in range(batch_size):
                j=randint(0, len(x_paths)-1)
                ### x
                x_path = x_paths[i]
                x_arr = x_path2arr(x_path)
                batch_x.append(x_arr)
                ### y
                y_path = y_paths[i]
                y_arr = y_path2arr(y_path)
                batch_y.append(y_arr)
                
            batch_x = np.concatenate(batch_x, axis=0)
            batch_y = np.concatenate(batch_y, axis=0)
            yield batch_x, batch_y
            
    def setup_base_dir(self):
        opt = self.opt
        source = 'domain_%s-k_%s-weightKNN_%s'%(opt.domain, opt.k, opt.weightKNN)
        self.base_dir = os.path.join(home, 'station2grid', 'datasets', 'npy', opt.domain, source)
    
    def setup_i_stations(self):
        val_stations = self.opt.val_stations.split('_')
        info = pd.read_csv(os.path.join(home,'station2grid','datasets','info','epa-station-info.csv'))
        c=~info.SiteEngName.isin(val_stations)
        self.i_stations = info[c].index.tolist()
                
    def setup_i_features(self, which_domain):
        domain_features = pd.read_csv(os.path.join(home,'station2grid','datasets','info','%s-features.csv'%(which_domain))).feature.tolist()
        features = self.opt.features.split('_')
        self.i_features = [domain_features.index(feature) for feature in features]
    
    def get_paths(self, file_name):
        opt=self.opt
        if file_name == 'grid': 
            return glob(os.path.join(self.base_dir, file_name, '*'))
        elif file_name == 'station': 
            return glob(os.path.join(self.base_dir, file_name, '*'))
        elif file_name == 'code':
            return glob(os.path.join(self.base_dir, file_name, opt.ae_type, '*'))
    
    def grid_path2arr(self, path):
        arr = np.load(path)
        arr = arr[...,:1]
        arr = self.normalize(arr)
        return arr
    
    def station_path2arr(self, path):
        arr = np.load(path)
        arr = arr[:, self.i_stations, :][..., self.i_features]
        arr = self.normalize(arr)
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
        arr = norm_const*arr
        arr = arr.astype('float')
        return arr
        