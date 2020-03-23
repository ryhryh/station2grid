from models.base_model import ModelBase
from models import networks
from tools.datasets import *
import os
import numpy as np

class ModelG2C(ModelBase):
    def __init__(self, opt):
        self.opt = opt
        self.setup()
        
    def setup(self):
        self.data = DataG2C(self.opt)
        self.autoencoder = networks.Autoencoder(self.opt)###
        self.set_group()
        self.set_path()
        self.model = self.autoencoder.define_autoencoder()
        
    def test(self): 
        print('testing')        
        weight = self.weight_path
        encoder = self.autoencoder.get_encoder(weight)
        
        code_dir = os.path.join(self.data.base_dir, 'code', self.opt.ae_type)
        os.makedirs(code_dir, exist_ok=True)
        
        grid_paths = self.data.x_paths
        for grid_path in grid_paths[:]:
            grid = self.data.grid_path2arr(grid_path)
            code = encoder.predict(grid)
            code_name = grid_path.split('/')[-1].replace('grid', 'code')
            code_path = os.path.join(code_dir, code_name)
            np.save(code_path, code)
            print(code_path) ###
        print('finish!')
        