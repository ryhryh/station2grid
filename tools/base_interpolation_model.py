import os
from abc import ABC, abstractmethod
import numpy as np
import itertools

import sys
home=os.path.expanduser("~")
sys.path.append(os.path.join(home, 'station2grid'))

from tools import CommonObj


class BaseInterpolationModel(ABC):
    def __init__(self, 
                 data,
                 lat_min=21.87, lat_max=25.34,
                 lon_min=120, lon_max=122.03,
                 step=0.01):
        
        self.data = data
        self.lat_min, self.lat_max = lat_min, lat_max
        self.lon_min, self.lon_max = lon_min, lon_max
        self.step = step
        self.lats = self.get_floatRange(self.lat_max, self.lat_min, -self.step)
        self.lons = self.get_floatRange(self.lon_min, self.lon_max, self.step)
        self.lat_num, self.lon_num = len(self.lats), len(self.lons)
        self.coordinates = self.get_coordinates()
        self.epa_station_info = CommonObj().epa_station_info
        self.epa_station_path = '/media/disk3/feynman52/station2grid/datasets/npy/epa/epa2019.npy'
        self.val_stations = self.data.opt.val_stations.split('_')
        
    def get_floatRange(self, start, stop, step):
        return np.arange(start, stop+step/10, step)
    
    def get_coordinates(self):
        coordinates = np.array(list(itertools.product(self.lats, self.lons)))
        return coordinates
    
    @abstractmethod
    def get_value(self):
        pass
    
    @abstractmethod
    def get_grid(self):
        pass
    
    
    @abstractmethod
    def get_values(self):
        pass
    
    @abstractmethod
    def get_grids(self):
        pass