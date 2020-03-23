from models import base_model
from models import networks ###
from tools import datasets


class ModelS2C(base_model.ModelBase):
    def __init__(self, opt):
        self.opt = opt
        self.setup()
        
    def setup(self):
        self.data = datasets.DataS2C(self.opt)
        self.dnn = networks.DNN(self.opt)###
        self.set_group()
        self.set_path()
        self.model = self.dnn.define_dnn()
        
        