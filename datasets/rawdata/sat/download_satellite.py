from ecmwfapi import ECMWFDataServer
from calendar import monthrange

def get_date(year,month):
    lastday=monthrange(year, month)[-1]
    start='%s-%02d-01'%(year,month)
    end='%s-%02d-%s'%(year,month,lastday)
    date="%s/to/%s"%(start,end)
    return date

def get_satellite_data(year,month):  
    time = "00:00:00" if year in [2015,2016] else "00:00:00/12:00:00"
    
    area_n = 25.4
    area_s = 21.8
    area_e = 122
    area_w = 120
    latlon = "%s/%s/%s/%s"%(area_n,area_w,area_s,area_e),
    
    file_path = '%s_%s.nc2'%(year,month)
    
    date = get_date(year,month)
    print(file_path,date,latlon,time)
    server = ECMWFDataServer()
    server.retrieve({
        "date": date,
        "time": time,
        "area": latlon,
        "target": file_path,
        "param": "73.210/74.210/167.128", # pm25/pm10/temperature
        "step": "3/6/9/12",
        "grid": "0.125/0.125",
        'format' : "netcdf",
        "class": "mc",
        "dataset": "cams_nrealtime",
        "expver": "0001",
        "levtype": "sfc",
        "stream": "oper",
        "type": "fc",
    })

    
if __name__ == '__main__':
    for year in range(2015,2019+1):
        for month in range(1,12+1):
            get_satellite_data(year,month)
    
    
    
    
