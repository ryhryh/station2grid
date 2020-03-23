import os
import sys
home=os.path.expanduser("~")
sys.path.append(os.path.join(home, 'station2grid'))
from tools.base_interpolation_model import BaseInterpolationModel

import numpy as np
from sklearn.neighbors import KNeighborsRegressor


class KnnModel(BaseInterpolationModel): 
    def __init__(self, data):
        super().__init__(data)
    
    def get_value(self, k, weightKNN, x_train, y_train, x_test):
        model_knn = KNeighborsRegressor(n_neighbors=k, weights=weightKNN)
        model_knn.fit(x_train, y_train)
        y_test = model_knn.predict(x_test) 
        return y_test 
    
    def get_grid(self, k, weightKNN, x_train, y_train):
        x_test = self.coordinates
        y_test = self.get_value(k, weightKNN, x_train, y_train, x_test)
        grid = y_test.reshape(1, self.lat_num, self.lon_num, -1)
        return grid
    
            
    def get_values(self, k, weightKNN):
        values = []
        for i in range(len(self.data.x_raw_select)): ###
            station = self.data.x_raw_select[i] ###
            x_train = self.data.train_info[['lat','lon']].values
            y_train = station.reshape(-1)
            x_test = self.data.valid_info[['lat','lon']].values
            value = self.get_value(k, weightKNN, x_train, y_train, x_test).reshape(1,-1,1)
            values.append(value)
        values = np.concatenate(values,axis=0)
        return values
    
    def get_grids(self, k, weightKNN):
        grids = []
        for i in range(len(self.data.x_raw_select)): ###
            station = self.data.x_raw_select[i] ###
            x_train = self.data.train_info[['lat','lon']].values
            y_train = station.reshape(-1)
            grid = self.get_grid(k, weightKNN, x_train, y_train)
            grids.append(grid)
        grids = np.concatenate(grids,axis=0)
        return grids