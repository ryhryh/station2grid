import os
import argparse
home=os.path.expanduser("~")

##########################################################################################  
########################################################################################## 
class OptionG2C():
    def __init__(self, model_name='grid2code', domain='air', k=5, weightKNN='distance', 
                 batch_size=10, epochs=5, ae_type='code_length-4576'):
        self.model_name = model_name
        self.domain = domain
        self.k = k
        self.weightKNN = weightKNN
        self.batch_size = batch_size
        self.epochs = epochs
        self.ae_type = ae_type
        
class OptionS2C():
    def __init__(self, model_name='station2code', features='pm25', val_stations='Chaozhou', 
                domain='air', k=5, weightKNN='distance', batch_size=10, epochs=5, 
                ae_type='code_length-4576', dnn_type='embedding-0~bn-1'):
        self.model_name = model_name
        self.features = features
        self.val_stations = val_stations
        self.domain = domain
        self.k = k
        self.weightKNN = weightKNN
        self.batch_size = batch_size
        self.epochs = epochs
        self.ae_type = ae_type
        self.dnn_type = dnn_type
        
        
class OptionS2GSD():
    def __init__(self, model_name='station2gridSD', features='pm25', val_stations='Chaozhou', 
                 domain='air', k=5, weightKNN='distance', 
                 ae_type='code_length-4576', dnn_type='embedding-0~bn-1'):
        self.model_name = model_name
        self.features = features
        self.val_stations = val_stations
        self.domain = domain
        self.k = k
        self.weightKNN = weightKNN
        self.ae_type = ae_type
        self.dnn_type = dnn_type
            
class OptionS2GMD():
    def __init__(self, model_name='station2gridMD', features='pm25', val_stations='Chaozhou', 
                 batch_size=10, epochs=5, ae_type='code_length-4576', dnn_type='embedding-0~bn-1',
                 domains='air_3_distance~sat_3_distance', composite_type='composite-conv~filter-8x8'):
        self.model_name = model_name
        self.features = features
        self.val_stations = val_stations
        self.batch_size = batch_size
        self.epochs = epochs
        self.ae_type = ae_type
        self.dnn_type = dnn_type
        self.domains = domains
        self.composite_type = composite_type

'''     
##########################################################################################  
########################################################################################## 
class BaseOptions():
    def __init__(self,fileName):
        self.fileName=fileName
        self.parser=argparse.ArgumentParser(description='base options...')
        self.initialize(self.parser)
        self.opt=self.parser.parse_args()
                
    def initialize(self,parser):
        parser.add_argument('--k', type=int, default=3)
        parser.add_argument('--weightKNN', type=str, default='uniform')
        
        if 'csv2npy' in self.fileName:
            parser.add_argument('--csv_path', type=str, default='')
            parser.add_argument('--thresholdKNN', type=int, default=1)
        elif 'grid2code' in self.fileName:
            parser.add_argument('--batch_size', type=int, default=2)
            parser.add_argument('--n_epochs', type=int, default=10)
            parser.add_argument('--domain', type=str, default='air')
            parser.add_argument('--autoencoderArcht', type=str, default='A1')
        elif 'station2code' in self.fileName:
            parser.add_argument('--batch_size', type=int, default=2)
            parser.add_argument('--n_epochs', type=int, default=10)
            parser.add_argument('--domain', type=str, default='air')
            parser.add_argument('--autoencoderArcht', type=str, default='A1')
            
            parser.add_argument('--codeLength', type=int, default=4576)
            parser.add_argument('--features', type=str, default='pm25_pm10')
            parser.add_argument('--valStations', type=str, default='Tamsui_Shilin')
        elif 'station2grid' in self.fileName:
            parser.add_argument('--domain', type=str, default='air')
            parser.add_argument('--autoencoderArcht', type=str, default='A1')
            
            parser.add_argument('--codeLength', type=int, default=4576)
            parser.add_argument('--features', type=str, default='pm25_pm10')
            parser.add_argument('--valStations', type=str, default='Tamsui_Shilin')
        
       '''