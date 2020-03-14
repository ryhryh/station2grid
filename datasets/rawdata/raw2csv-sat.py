import pandas as pd
import os
import numpy as np
import netCDF4

def get_one_month(year,month):
    print(year,month)
    path=os.path.join(os.path.expanduser("~"),'station2grid','datasets','rawdata','sat','%s_%s.nc'%(year,month))
    dataset = netCDF4.Dataset(path)
    lat_arr=np.array(dataset.variables['latitude'][:]) 
    lon_arr=np.array(dataset.variables['longitude'][:]) 
    pm25_arr=np.array(dataset.variables['pm2p5'][:]) 
    time_var = dataset.variables['time']
    time_arr = netCDF4.num2date(time_var[:],time_var.units)
    time_arr = np.array([pd.to_datetime(t+pd.Timedelta(hours=8)) for t in time_arr]) 
    
    one_month=pd.DataFrame()
    for i_dt,dt in enumerate(time_arr[:10]): ######
        for i_lat,lat in enumerate(lat_arr):
            for i_lon,lon in enumerate(lon_arr):
                row=pd.DataFrame()
                row['dt']=[dt]
                row['lat']=[round(float(lat),3)]
                row['lon']=[round(float(lon),3)]
                row['pm25']=[round(pm25_arr[i_dt,i_lat,i_lon]*10**9,3)]
                one_month=one_month.append(row)
    
    # include [11,14,17] exclude others
    c=one_month.dt.apply(lambda x: x.hour in [11,14,17]) 
    one_month_removeAbnormal=one_month[c]
    
    return one_month_removeAbnormal

if __name__=='__main__':
    print('processing satellite...')
    ##########################################################################################
    dfs=[get_one_month(year,month) for year in range(2015,2018+1)[:] for month in range(1,12+1)[:1]]
    df=pd.concat(dfs,axis=0)
                   
    csv_path=os.path.join(os.path.expanduser("~"),'station2grid','datasets','csv','sat.csv')
    df.to_csv(csv_path,index=False)
    ##########################################################################################
    print('finish satellite!')
          