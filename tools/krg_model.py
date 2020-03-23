import os
import sys
home=os.path.expanduser("~")
sys.path.append(os.path.join(home, 'station2grid'))
from tools.base_interpolation_model import BaseInterpolationModel

import numpy as np
from pykrige.ok import OrdinaryKriging


class KrgModel(BaseInterpolationModel): 
    def __init__(self, data):
        super().__init__(data)
        
    def get_value(self, x_train, y_train, x_test):
        OK = OrdinaryKriging(
            x_train[:,0], 
            x_train[:,1], 
            y_train, 
            variogram_model='linear',
            verbose=0, 
            enable_plotting=0
        )
        y_test, ss = OK.execute('points', x_test[:,0], x_test[:,1])
        return y_test
    
    def get_grid(self, x_train, y_train):
        x_test = self.coordinates
        y_test = self.get_value(x_train, y_train, x_test)
        grid = y_test.reshape(1, self.lat_num, self.lon_num, -1)
        return grid
    
    def get_values(self):
        values = []
        for i in range(len(self.data.x_raw_select)): ###
            #print(i)
            station = self.data.x_raw_select[i] ###
            x_train = self.data.train_info[['lat','lon']].values
            y_train = station.reshape(-1)
            x_test = self.data.valid_info[['lat','lon']].values
            value = self.get_value(x_train, y_train, x_test).reshape(1,-1,1)
            values.append(value)
        values = np.concatenate(values,axis=0)
        return values
    
    def get_grids(self):
        grids = []
        for i in range(len(self.data.x_raw_select)): ###
            station = self.data.x_raw_select[i] ###
            x_train = self.data.train_info[['lat','lon']].values
            y_train = station.reshape(-1)
            grid = self.get_grid(x_train, y_train)
            grids.append(grid)
        grids = np.concatenate(grids,axis=0)
        return grids