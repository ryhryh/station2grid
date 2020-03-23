import pandas as pd
import numpy as np
import os
home=os.path.expanduser("~")

#import sys 
#sys.path.append(os.path.join(home, 'station2grid'))
from tools.knn_model import KnnModel


def get_oneYear(year):
#     csv_path=os.path.join(home,'station2grid','datasets','csv','csv_flies','epa%s.csv'%(year))
    csv_path=os.path.join(home,'station2grid','datasets','csv','csv_files','epa%s.csv'%(year))
    oneYear=pd.read_csv(csv_path)
    oneYear=pd.merge(left=oneYear,
                     right=epaStationInfo[['SiteName','lat','lon']],
                     left_on='station',
                     right_on='SiteName',
                     how='left')
    oneYear=oneYear[['dt','lat','lon','station','feature','value']]
    
    exclude_features = ['WD_HR', 'WS_HR'] ###
    oneYear=oneYear[~oneYear.feature.isin(exclude_features)]
    oneYear=oneYear.replace(
        ['PM2.5','PH_RAIN','RAIN_COND','AMB_TEMP'], 
        ['pm25','PHRAIN','RAINCOND','AMBTEMP'])

    return oneYear

def get_oneDtArr(oneDt,features):
    oneDtArr=[]
    for feature in features:
        oneDtFea=oneDt[oneDt.feature==feature]
        x_train=oneDtFea[['lat','lon']].values
        y_train=oneDtFea.value.values
        k,weightKNN=1,'distance'
        x_test=epaStationInfo[['lat','lon']].values
        y_test=model.get_value(k, weightKNN, x_train, y_train, x_test)
        y_test=y_test.reshape(-1,1)
        oneDtArr.append(y_test)
    oneDtArr=np.concatenate(oneDtArr,axis=-1)    
    
    windArr=get_windArr(oneDt)
    oneDtArr=np.concatenate([oneDtArr,windArr],axis=-1) 
    
    oneDtArr=oneDtArr.reshape((1,)+oneDtArr.shape)
    return oneDtArr

def get_windArr(oneDt):
    speed=oneDt[oneDt.feature=='WIND_SPEED']
    direc=oneDt[oneDt.feature=='WIND_DIREC']
    wind=pd.merge(left=speed,right=direc,on=['dt','lat','lon','station'],how='inner')
    wind['wind_cos']=wind.apply(lambda row: (row.value_x)*(np.cos(np.deg2rad(row.value_y))) ,axis=1)
    wind['wind_sin']=wind.apply(lambda row: (row.value_x)*(np.sin(np.deg2rad(row.value_y))) ,axis=1)
    k=1
    weightKNN='distance'
    x_train=wind[['lat','lon']].values
    y_train=wind[['wind_cos','wind_sin']].values
    x_test=epaStationInfo[['lat','lon']].values
    y_test=model.get_value(k, weightKNN, x_train, y_train, x_test)
    return y_test

def saveFeatures(features):
    path=os.path.join(home,'station2grid','datasets','info','epa-features.csv')
    dummy=pd.DataFrame({'feature':features})
    dummy.to_csv(path)

def get_allDtArr(oneYear):
    dts=sorted(oneYear.dt.unique()) ### unique預設沒有sort.................
    
    features=[
    'pm25', 'PM10', 'AMBTEMP', 'RH', 'CO', 'NO', 'NO2', 'NOx', 'O3', 'PHRAIN', 
    'RAINFALL', 'RAINCOND', 'SO2', 'CH4', 'NMHC', 'THC', 'UVB'
    ]
    
    features_=features+['WINDCOS', 'WINDSIN']
    saveFeatures(features_)
    
    allDtArr=[]
    ######################################################################
    for i,dt in enumerate(dts[:]):
        print(i,len(dts))
        oneDt=oneYear[oneYear.dt==dt]
        oneDtArr=get_oneDtArr(oneDt,features)
        allDtArr.append(oneDtArr)
    ######################################################################
    allDtArr=np.concatenate(allDtArr,axis=0)   
    return allDtArr


if __name__=='__main__':
    
    print('processing...') 
    ############################################################################
    model = KnnModel()
    epaStationInfo = pd.read_csv(os.path.join(home, 'station2grid', 'datasets', 'info', 'epa-station-info.csv'))

    dir_path = os.path.join(os.path.expanduser("~"), 'station2grid', 'datasets', 'npy', 'epa')
    os.makedirs(dir_path, exist_ok=True)
    
    for year in [2014,2015,2016,2017,2018,2019]: #####################
        print(year)
        oneYear = get_oneYear(year)
        allDtArr = get_allDtArr(oneYear) 
        npy_path = os.path.join(dir_path, 'epa%s'%(year))
        np.save(npy_path,allDtArr)
    ############################################################################
    print('finish!') 
