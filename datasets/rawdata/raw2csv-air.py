import pandas as pd
import os

def change_time_interval(dt,time_interval):
    dt=pd.to_datetime(dt)
    minute = dt.minute
    second = dt.second
    minute = minute+(second/60)
    minute = round(minute/time_interval) * time_interval
    dt = dt.replace(minute=0,second=0)
    dt = dt+pd.Timedelta(minutes=minute)
    return dt

def get_one_year(year):  
    print(year)
    path=os.path.join(os.path.expanduser("~"),'station2grid','datasets','rawdata','air','airbox-%s.csv'%(year))
    df=pd.read_csv(path)
    #df=df[:100000].copy() ###############################################################
    
    df['dt']=df.apply(lambda row: row['Date']+' '+row['Time'],axis=1)
    df['dt2']=df['dt'].apply(lambda x: change_time_interval(x,30))
    df_median=df.groupby(by=['dt2','lat','lon']).median().reset_index(drop=False)
    df_median=df_median.drop(columns=['PM1'])
    columns={
        'dt2':'dt',
        'PM2.5':'pm25',
        'PM10':'pm10',
        'Temperature':'temperature',
        'Humidity':'humidity',
    }
    df_median=df_median.rename(columns=columns)
    
    # remove abnormal
    pm25_threshold=350
#     abnorm_locs=df_median[df_median.pm25>pm25_threshold].drop_duplicates(subset=['lat','lon'])
#     abnorm_locs=abnorm_locs[['lat','lon']].values
#     c=df_median.apply(lambda row: (row.lat,row.lon) not in abnorm_locs,axis=1)
#     df_removeAbnormal=df_median[c]
    
    c=df_median.apply(lambda row: row.pm25<pm25_threshold, axis=1)
    df_removeAbnormal=df_median[c]

    return df_removeAbnormal


if __name__=='__main__':
    print('processing airbox...')
    ##########################################################################################
    dfs=[get_one_year(year) for year in [2017,2018]]            
    df=pd.concat(dfs,axis=0)
                   
    csv_path=os.path.join(os.path.expanduser("~"),'station2grid','datasets','csv','air2.csv') ###
    df.to_csv(csv_path,index=False)
    ##########################################################################################
    print('finish airbox!')
          