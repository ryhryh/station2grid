from models.networks import AE
from tools.data_generator import DataGenerator
import pandas as pd
import os
from glob import glob
import numpy as np
home=os.path.expanduser("~")

class Grid2Code():
    def __init__(self, opt):
        self.opt = opt
        self.ae = AE(opt)
        self.dataGenerator = DataGenerator(opt)
        self.setup_weight_dir()
    
    def train(self):
        print('training...')
        dataGenerator = self.dataGenerator
        g_train = dataGenerator.generator_train
        g_valid = dataGenerator.generator_valid
        
        self.autoencoder = self.ae.define_ae()
        
        callbacks = self.ae.get_callbacks(self.weight_dir)
        
        history = self.autoencoder.fit_generator(
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
        
    def test(self): 
        print('testing...')
        dataGenerator = self.dataGenerator
        weights = sorted(glob(os.path.join(self.weight_dir, '*hdf5')))
        weight = weights[-1]
        
        encoder_best = self.ae.get_encoder(weight)
        
        code_dir = os.path.join(dataGenerator.base_dir, 'code', self.opt.ae_type)
        os.makedirs(code_dir, exist_ok=True)
        
        grid_paths = dataGenerator.get_paths('grid')
        for grid_path in grid_paths[:]:
            grid = dataGenerator.grid_path2arr(grid_path)
            code = encoder_best.predict(grid)
            code_name = grid_path.split('/')[-1].replace('grid', 'code')
            code_path = os.path.join(code_dir, code_name)
            np.save(code_path, code)
        print('finish!')
        
    
    def save_history(self, history):
        df_history = pd.DataFrame(history.history)
        path = os.path.join(self.weight_dir, 'history.csv',)
        df_history.to_csv(path, index=False)
        
    def setup_weight_dir(self):
        opt = self.opt
        source = 'domain_%s-k_%s-weightKNN_%s'%(opt.domain, opt.k, opt.weightKNN)
        self.weight_dir = os.path.join(home, 'station2grid', 'weights', 'single', source, opt.model_name, opt.ae_type)
        os.makedirs(self.weight_dir, exist_ok=True)
        
        