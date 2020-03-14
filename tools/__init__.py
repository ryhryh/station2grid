from sklearn.neighbors import KNeighborsRegressor
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import os
import pandas as pd
home=os.path.expanduser("~")

class CommonObj():
    def __init__(self):
        self.epa_station_info = pd.read_csv(os.path.join(home,'station2grid','datasets','info','epa-station-info.csv'))
        

class CustomKNN():
    def __init__(self,lat_min=21.87,lat_max=25.34,lon_min=120,lon_max=122.03,step=0.01):
        self.lat_min,self.lat_max=lat_min,lat_max
        self.lon_min,self.lon_max=lon_min,lon_max
        self.step=step
        self.lats=self.get_floatRange(self.lat_max,self.lat_min,-self.step)
        self.lons=self.get_floatRange(self.lon_min,self.lon_max,self.step)
        self.lat_num,self.lon_num=len(self.lats),len(self.lons)
        self.coordinates=self.get_coordinates()
        
    def get_epaStationInfo(self):
        return pd.read_csv(os.path.join(home,'station2grid','datasets','info','epa-station-info.csv'))

    def get_floatRange(self,start,stop,step):
        return np.arange(start,stop+step/10,step)
    
    def get_coordinates(self):
        coordinates=np.array(list(itertools.product(self.lats,self.lons)))
        return coordinates
    
    def get_knn(self,k,weightKNN,x_train,y_train,x_test):
        model_knn=KNeighborsRegressor(n_neighbors=k, weights=weightKNN)
        model_knn.fit(x_train,y_train)
        y_test=model_knn.predict(x_test) 
        return y_test 
    
    def get_knn_grid(self,k,weightKNN,x_train,y_train):
        x_test=self.coordinates
        model_knn=KNeighborsRegressor(n_neighbors=k, weights=weightKNN)
        model_knn.fit(x_train,y_train)
        y_test=model_knn.predict(x_test) 
        knn_grid=y_test.reshape(1,self.lat_num,self.lon_num,-1)
        return knn_grid
    

def plotMap(lats,lons,vals,title,fig_num,latMin=21,latMax=26,lonMin=119,lonMax=123,bar_min=0,bar_max=70):
    shrink=1/fig_num
    m = Basemap(projection='merc',
                llcrnrlat=latMin,
                urcrnrlat=latMax,
                llcrnrlon=lonMin,
                urcrnrlon=lonMax,
                lat_ts=20,
                resolution='i')
    m.drawcoastlines()
    m.drawcountries()
    
    parallels = [num for num in range(latMin,latMax)] # np.arange(21.,27.,1.0)
    m.drawparallels(parallels,labels=[1,0,0,0])
    meridians = [num for num in range(lonMin,lonMax)] # np.arange(118.,124.,1.0)
    m.drawmeridians(meridians,labels=[0,0,0,1])
    
    x,y = m(lons, lats)
    cmap = plt.cm.get_cmap('Reds')
    sc = plt.scatter(x,y,c=vals,
                     vmin=bar_min, 
                     vmax=bar_max,
                     cmap=cmap,
                     s=50,
                     edgecolors='none')
    cbar = plt.colorbar(sc, shrink = shrink)
    plt.title(title, fontsize=18)
   