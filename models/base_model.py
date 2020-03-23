from keras.callbacks import EarlyStopping, ModelCheckpoint

from abc import ABC, abstractmethod
from glob import glob
import pandas as pd
import os
home=os.path.expanduser("~")

import multiprocessing


class ModelBase(ABC):
    def __init__(self): pass
    
    @abstractmethod
    def setup(self): pass
    
    def train(self):
        print('training')
        g_train = self.data.g_train
        g_valid = self.data.g_valid
        callbacks = self.get_callbacks(min_delta=0, patience=20) ###
        
        self.history = self.model.fit_generator(
            generator = g_train,
            steps_per_epoch = (len(self.data.x_train_paths) // self.opt.batch_size),

            validation_data = g_valid,
            validation_steps = (len(self.data.x_valid_paths) // self.opt.batch_size),

            epochs = self.opt.epochs,
            verbose = 1, 
            callbacks = callbacks,

            use_multiprocessing = True,
            workers = multiprocessing.cpu_count(), # 32
            max_queue_size = 10,
        )
        
        self.save_history(self.history.history)
        print('finish!')
        
    def set_path(self):
        self.weight_path = os.path.join(
            home, 'station2grid', 'weights', self.group+'---'+'best_weight.hdf5')
        
        self.history_path = os.path.join(
            home, 'station2grid', 'weights', self.group+'---'+'history.csv')
    
    def set_group(self):
        dic = vars(self.opt)
        self.group = '--'.join(
            [key+'-'+str(dic[key]) for key in dic if key not in ['epochs', 'batch_size']] ###
        )
        
    def save_history(self, history):
        df_history = pd.DataFrame(history)
        df_history.to_csv(self.history_path, index=False)
    
    def get_callbacks(self, min_delta=0, patience=10):
        early_stopping = EarlyStopping(
            monitor='val_loss', mode='min', verbose=0, patience=patience, min_delta=min_delta) ### 

        checkpointer = ModelCheckpoint(
            filepath=self.weight_path, verbose=0, period=1, monitor='val_loss', 
            save_best_only=True, mode='min')

        callbacks = [early_stopping, checkpointer]
        return callbacks
        