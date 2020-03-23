import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Path, PathPatch

from sklearn.neighbors import KNeighborsRegressor
import itertools
import numpy as np
import os
import pandas as pd
home=os.path.expanduser("~")

class CommonObj():
    def __init__(self):
        self.epa_station_info = pd.read_csv(
            os.path.join(home,'station2grid','datasets','info','station-info'))
        
def plotMap(ax,
            lats, lons, vals,
            title, title_size=18,
            latMin=21, latMax=26, lonMin=119, lonMax=123,
            bar_min=0, bar_max=70):    
    
    m = Basemap(projection='merc',
                llcrnrlat=latMin, urcrnrlat=latMax, llcrnrlon=lonMin, urcrnrlon=lonMax,
                lat_ts=20, resolution='i', ax=ax)
    
    parallels = [num for num in range(latMin,latMax)] # np.arange(21.,27.,1.0)
    meridians = [num for num in range(lonMin,lonMax)] # np.arange(118.,124.,1.0)
    m.drawparallels(parallels,labels=[0,0,0,0],linewidth=0.1) # 1,0,0,0
    m.drawmeridians(meridians,labels=[0,0,0,0],linewidth=0.1) # 0,0,0,1
    m.drawcoastlines()
    m.drawcountries()
    
    x, y = m(lons, lats)
    cmap = plt.cm.get_cmap('Reds')
    sc = ax.scatter(x, y, c=vals, vmin=bar_min, vmax=bar_max, cmap=cmap, s=50, edgecolors='none')
    cbar = plt.colorbar(sc, ax=ax)
    ax.set_title(title, fontdict={'fontsize': title_size})
    
    # mask area outside Taiwan
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    map_edges = np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
    polys = [p.boundary for p in m.landpolygons]
    polys = [map_edges]+polys[:]
    codes = [ [Path.MOVETO]+[Path.LINETO for p in p[1:]] for p in polys]
    polys_lin = [v for p in polys for v in p]
    codes_lin = [xx for cs in codes for xx in cs]
    path = Path(polys_lin, codes_lin)
    patch = PathPatch(path,facecolor='white', lw=0)
    ax.add_patch(patch) 
    
    
def get_predict_result(y_true, y_hat, group, val_station_names=['Tainan','Taitung']):
        
    dts=pd.date_range(start='2019', end='2020', periods=None, freq='H', closed='left')

    a = pd.DataFrame(y_true)
    a.columns = val_station_names
    a['is_real'] = 'real'
    a['dt'] = dts

    b = pd.DataFrame(y_hat)
    b.columns = val_station_names
    b['is_real'] = 'predict'
    b['dt'] = dts

    c = pd.concat([a,b], axis=0)
    c['group'] = group
    
    c_melt = pd.melt(c, id_vars=['group', 'dt', 'is_real'], var_name='station', value_vars=val_station_names)

    c_pt = pd.pivot_table(
        c_melt, values='value', columns=['is_real'],
        index=['group', 'station', 'dt']).reset_index(drop=False)
    return c_pt        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
'''
def plotMap(ax,lats,lons,vals,title,fig_num,
            latMin=21,latMax=26,
            lonMin=119,lonMax=123,
            bar_min=0,bar_max=70):    
    
    shrink=1/fig_num
    m = Basemap(projection='merc',
                llcrnrlat=latMin,
                urcrnrlat=latMax,
                llcrnrlon=lonMin,
                urcrnrlon=lonMax,
                lat_ts=20,
                resolution='i')
    
    parallels = [num for num in range(latMin,latMax)] # np.arange(21.,27.,1.0)
    meridians = [num for num in range(lonMin,lonMax)] # np.arange(118.,124.,1.0)
    m.drawparallels(parallels,labels=[1,0,0,0],linewidth=0.1)
    m.drawmeridians(meridians,labels=[0,0,0,1],linewidth=0.1)
    m.drawcoastlines()
    m.drawcountries()
    
    x,y = m(lons, lats)
    cmap = plt.cm.get_cmap('Reds')
    sc = plt.scatter(x,y,c=vals,
                     vmin=bar_min, 
                     vmax=bar_max,
                     cmap=cmap,
                     s=50,
                     edgecolors='none')
    cbar = plt.colorbar(sc, shrink = shrink)
    plt.title(title, fontsize=18) #, fontsize=18
    
    # mask area outside Taiwan
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    map_edges = np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
    polys = [p.boundary for p in m.landpolygons]
    polys = [map_edges]+polys[:]
    codes = [ [Path.MOVETO]+[Path.LINETO for p in p[1:]] for p in polys]
    polys_lin = [v for p in polys for v in p]
    codes_lin = [xx for cs in codes for xx in cs]
    path = Path(polys_lin, codes_lin)
    patch = PathPatch(path,facecolor='white', lw=0)
    ax.add_patch(patch) 

# for knn,krg
def get_predict_result2(predict, data, group):
    valid_info = data.valid_info
    val_indexs = valid_info.index.values
    val_names = valid_info.SiteEngName.values
    
    real = data.x_raw[:, val_indexs, 0] ###
    
    dts=pd.date_range(start='2019', end='2020', periods=None, freq='H', closed='left')

    a = pd.DataFrame(real)
    a.columns = val_names
    a['is_real'] = 'real'
    a['dt'] = dts

    b = pd.DataFrame(predict)
    b.columns = val_names
    b['is_real'] = 'predict'
    b['dt'] = dts

    c = pd.concat([a,b], axis=0)
    c['group'] = group
    
    c_melt = pd.melt(c, id_vars=['group', 'dt', 'is_real'], var_name='station', value_vars=val_names)

    c_pt = pd.pivot_table(
        c_melt, values='value', columns=['is_real'],
        index=['group', 'station', 'dt']).reset_index(drop=False)
    return c_pt


def get_predict_result(grids, model):
    stations = model.data.x_raw
    valid_info = model.data.valid_info
    val_indexs = valid_info.index.values
    val_coors = valid_info[['row','col']].values
    val_names = valid_info.SiteEngName.values
    
    predict = grids[:, val_coors[:,0], val_coors[:,1], 0]
    real = stations[:, val_indexs, 0]
    
    dts=pd.date_range(start='2019', end='2020', periods=None, freq='H', closed='left')

    a = pd.DataFrame(real)
    a.columns = val_names
    a['is_real'] = 'real'
    a['dt'] = dts

    b = pd.DataFrame(predict)
    b.columns = val_names
    b['is_real'] = 'predict'
    b['dt'] = dts

    c = pd.concat([a,b], axis=0)
    c['group'] = model.group
    
    c_melt = pd.melt(c, id_vars=['group', 'dt', 'is_real'], var_name='station', value_vars=val_names)

    c_pt = pd.pivot_table(
        c_melt, values='value', columns=['is_real'],
        index=['group', 'station', 'dt']).reset_index(drop=False)
    return c_pt'''