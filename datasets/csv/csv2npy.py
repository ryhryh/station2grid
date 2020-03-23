import pandas as pd
import numpy as np
import os
import argparse

#import sys 
#sys.path.append(os.path.join(os.path.expanduser("~"), 'station2grid'))
from tools.knn_model import KnnModel
from tools.options import BaseOptions


class Csv2npy:
    def __init__(self, csv_path, k, weightKNN, threshold):
        self.home = os.path.expanduser("~")
        self.epaStationInfo = pd.read_csv(os.path.join(self.home, 'station2grid', 'datasets', 'info', 'epa-station-info.csv'))
        self.csv_path = csv_path
        self.domain = csv_path.split('/')[-1].split('.')[0]
        self.single = 'domain_%s-k_%s-weightKNN_%s'%(self.domain, k, weightKNN) ##########
        self.k = k
        self.weightKNN = weightKNN
        #self.customKNN=CustomKNN()
        self.threshold = threshold
        
    def saveFeatures(self,features):
        path=os.path.join(self.home, 'station2grid', 'datasets', 'info', '%s-features.csv'%(self.domain))
        dummy = pd.DataFrame({'feature':features})
        dummy.to_csv(path)
    
    def get_grid_station(self, one_dt, k, weightKNN):
        #customKNN=self.customKNN
        knnModel = KnnModel()
        x_train = one_dt[['lat', 'lon']]
        y_train = one_dt.iloc[:, 3:]
        #knn_grid=customKNN.get_knn_grid(k,weightKNN,x_train,y_train)
        knn_grid = knnModel.get_grid(k, weightKNN, x_train, y_train)
        
        grid_pm25 = knn_grid[...,:1] ###
        station = knn_grid[:, self.epaStationInfo.row, self.epaStationInfo.col, :]
        return grid_pm25,station

    def alldt2npy(self):
        domain = self.domain
        home = self.home
        single = self.single
        
        df = pd.read_csv(self.csv_path)
        
        # save features for single domain
        features = df.columns[3:].tolist()
        self.saveFeatures(features)

        # make dir for (grid,station)
        grid_dir = os.path.join(home, 'station2grid', 'datasets', 'npy', domain, single, 'grid')
        station_dir = os.path.join(home, 'station2grid', 'datasets', 'npy', domain, single, 'station')
        os.makedirs(grid_dir, exist_ok=True)
        os.makedirs(station_dir, exist_ok=True)
        
        dts = df.dt.unique()
        for dt in dts[:]: ###############################################################
        
        #for i in [1000,2000,3000]:
            #dt = dts[i]
            
            one_dt=df[df.dt==dt]
            
            # if number of sources < threshold, pass
            print(dt, len(one_dt)) ###
            if len(one_dt) < self.threshold: 
                print('pass') ###
                continue
                
            grid_pm25, station = self.get_grid_station(one_dt, self.k, self.weightKNN)
            # save npy (grid,station)
            dt_str = str(dt)[:19]
            np.save(os.path.join(grid_dir, dt_str+'_grid'), grid_pm25)
            np.save(os.path.join(station_dir, dt_str+'_station'), station)


if __name__=='__main__':
    
    fileName = os.path.basename(__file__)
    baseOptions = BaseOptions(fileName)
    args = baseOptions.opt
    
    csv_path = args.csv_path
    k = args.k
    weightKNN = args.weightKNN
    thresholdKNN = args.thresholdKNN

    csv2npy = Csv2npy(csv_path, k, weightKNN, thresholdKNN)
    print('processing %s...'%(csv2npy.single))
    csv2npy.alldt2npy()
    print('finish!') 
