from models.networks import AE, FCNN
from tools.data_generator import DataGenerator
import pandas as pd
import os
from glob import glob
import numpy as np
home=os.path.expanduser("~")

class Station2Code():
    def __init__(self, opt):
        self.opt = opt
        self.fcnn = FCNN(opt)
        self.dataGenerator = DataGenerator(opt)
        
        self.n_val_stations = len(self.opt.val_stations.split('_')) ###
        self.n_features = len(self.opt.features.split('_')) ###
        self.setup_weight_dir()
    
    def train(self):
        print('training...')
        dataGenerator = self.dataGenerator
        g_train = dataGenerator.generator_train
        g_valid = dataGenerator.generator_valid
        
        self.s2c_model = self.fcnn.define_fcnn(self.n_val_stations, self.n_features, self.opt.code_length)
        
        callbacks = self.fcnn.get_callbacks(self.weight_dir)
        
        history = self.s2c_model.fit_generator(
            generator = g_train,
            steps_per_epoch = (len(dataGenerator.x_train_paths) // self.opt.batch_size),

            validation_data = g_valid,
            validation_steps = (len(dataGenerator.x_valid_paths) // self.opt.batch_size),

            epochs = self.opt.n_epochs,
            verbose = 0,
            callbacks = callbacks,

            use_multiprocessing = True,
            workers = 8,
            max_queue_size = 10,
        )
        
        self.save_history(history)
        print('finish!')
        
    
    def save_history(self, history):
        df_history = pd.DataFrame(history.history)
        path = os.path.join(self.weight_dir, 'history.csv',)
        df_history.to_csv(path, index=False)
        
    def setup_weight_dir(self):
        opt = self.opt
        source = 'domain_%s-k_%s-weightKNN_%s'%(opt.domain, opt.k, opt.weightKNN)
        self.weight_dir = os.path.join(home, 'station2grid', 'weights', 'single', source, opt.model_name, opt.ae_type, str(self.n_val_stations), opt.val_stations, opt.features)
        os.makedirs(self.weight_dir, exist_ok=True)
        
        