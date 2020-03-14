import pandas as pd
import numpy as np
import os
home=os.path.expanduser("~")

import sys 
sys.path.append(os.path.join(home,'station2grid'))
from tools import CustomKNN

def get_oneYear(year):
    csv_path=os.path.join(home,'station2grid','datasets','csv','epa','epa%s.csv'%(year))
    oneYear=pd.read_csv(csv_path)
    oneYear=pd.merge(left=oneYear,
                     right=epaStationInfo[['SiteName','lat','lon']],
                     left_on='station',
                     right_on='SiteName',
                     how='left')
    oneYear=oneYear[['dt','lat','lon','station','feature','value']]
    return oneYear

def get_oneDtArr(oneDt,features):
    oneDtArr=[]
    for feature in features:
        oneDtFea=oneDt[oneDt.feature==feature]
        x_train=oneDtFea[['lat','lon']].values
        y_train=oneDtFea.value.values
        k,weightKNN=1,'distance'
        x_test=epaStationInfo[['lat','lon']].values
        y_test=customKNN.get_knn(k,weightKNN,x_train,y_train,x_test)
        y_test=y_test.reshape(1,-1,1)
        oneDtArr.append(y_test)
    oneDtArr=np.concatenate(oneDtArr,axis=-1)    
    return oneDtArr

def saveFeatures(features):
    path=os.path.join(home,'station2grid','datasets','info','epa-features.csv')
    dummy=pd.DataFrame({'feature':features})
    dummy.to_csv(path)

def get_allDtArr(oneYear):
    dts=oneYear.dt.unique()
    features=oneYear.feature.unique()
    
    featureDic={'AMB_TEMP':'temperature', 
                'PM10':'pm10', 
                'PM2.5':'pm25', 
                'RH':'humidity'}
    features_=[featureDic[fea] for fea in features]
    saveFeatures(features_)
    
    allDtArr=[]
    for i,dt in enumerate(dts[:]):
        oneDt=oneYear[oneYear.dt==dt]
        if len(oneDt.feature.unique())<len(features): continue
        
        oneDtArr=get_oneDtArr(oneDt,features)
        allDtArr.append(oneDtArr)
        
    allDtArr=np.concatenate(allDtArr,axis=0)
    return allDtArr


if __name__=='__main__':
    
    print('processing...') 
    ############################################################################
    
    customKNN=CustomKNN()
    epaStationInfo=pd.read_csv(os.path.join(home,'station2grid','datasets','info','epa-station-info.csv'))

    dir_path=os.path.join(os.path.expanduser("~"),'station2grid','datasets','npy','epa')
    os.makedirs(dir_path,exist_ok=True)
    
    for year in range(2014,2019+1)[:3]:
        print(year)
        oneYear=get_oneYear(year)
        allDtArr=get_allDtArr(oneYear) 
        npy_path=os.path.join(dir_path,'epa%s'%(year))
        np.save(npy_path,allDtArr)
    ############################################################################
    print('finish!') 
