import os
import argparse
home=os.path.expanduser("~")

class Opt():
    def __init__(self, domain='air', k=5, weightKNN='distance', ae_type='A1', batch_size=2, n_epochs=10, model_name='grid2code', features='pm25', val_stations='Shilin_Guting', code_length=4576, epa_station_path=''):
        self.domain = domain
        self.k = k
        self.weightKNN = weightKNN
        self.ae_type = ae_type
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.model_name = model_name
        self.features = features
        self.val_stations = val_stations
        self.code_length = code_length
        self.epa_station_path = epa_station_path
              
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
        
       