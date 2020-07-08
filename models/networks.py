from keras.layers import Input, Flatten, Dense, Conv2D, MaxPooling2D, UpSampling2D, \
BatchNormalization, Activation, Lambda, Concatenate, Reshape

from keras.models import Model, load_model
from keras import backend as K

import re
import pandas as pd
from glob import glob
import os

from models import station2gridSD_model
from tools import options, CommonObj

home = os.path.expanduser("~")
info = CommonObj().epa_station_info
n_all_stations = len(info)

#########################################################################################################
#########################################################################################################
class DNN():
    def __init__(self, opt):
        self.opt = opt
        self.setup()
    
    def setup(self):
        self.n_val_stations = len(self.opt.val_stations.split('_')) 
        self.n_features = len(self.opt.features.split('_')) 
        self.parse()
    
    def parse(self):
        key = 'code_length'
        self.code_length = int(re.search('(~|^)%s-(.+?)(~|$)'%(key), self.opt.ae_type).group(2))
        
    def define_dnn(self):
        num_station = n_all_stations - self.n_val_stations
        
        input_ = Input(shape=(num_station, self.n_features))
         
        if self.opt.dnn_type == 'a1':
            x = Flatten()(input_)
            x = Dense(300, activation='relu')(x) 
            
        output_ = Dense(self.code_length, activation='linear')(x)
        
        model = Model(input_, output_)
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'mse'])
        model.summary() ###
        return model
    
    def get_dnn(self, weight):
        dnn = load_model(weight)
        return dnn

#########################################################################################################
#########################################################################################################
class Autoencoder():
    def __init__(self, opt): 
        self.opt = opt
        
    def define_autoencoder(self):
        ae_type = self.opt.ae_type
        if ae_type == 'code_length-4576':
            autoencoder = self.code_length_4576()
        elif ae_type == 'code_length-2288': 
            autoencoder = self.code_length_2288()
        elif ae_type == 'code_length-3432':
            autoencoder = self.code_length_3432()
        elif ae_type == 'code_length-5720':
            autoencoder = self.code_length_5720()
        elif ae_type == 'code_length-6864':
            autoencoder = self.code_length_6864()    
        
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        autoencoder.summary()
        return autoencoder
    
    def code_length_4576(self,):
        input_shape = (348, 204, 1)
        input_img = Input(shape=input_shape)  
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(4, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(4, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='relu')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        autoencoder = Model(input_img, decoded)
        return autoencoder
    
    def code_length_6864(self,):
        input_shape = (348, 204, 1)
        input_img = Input(shape=input_shape)  
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(6, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(6, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='relu')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        autoencoder = Model(input_img, decoded)
        return autoencoder
    
    def code_length_5720(self,):
        input_shape = (348, 204, 1)
        input_img = Input(shape=input_shape)  
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(5, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(5, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='relu')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        autoencoder = Model(input_img, decoded)
        return autoencoder
    
    def code_length_3432(self,):
        input_shape = (348, 204, 1)
        input_img = Input(shape=input_shape)  
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(3, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(3, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='relu')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        autoencoder = Model(input_img, decoded)
        return autoencoder
    
    def code_length_2288(self,):
        input_shape = (348, 204, 1)
        input_img = Input(shape=input_shape)  
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(2, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(2, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='relu')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        autoencoder = Model(input_img, decoded)
        return autoencoder
    
    def get_encoder(self, weight): 
        autoencoder = load_model(weight)
        encoder = Model(autoencoder.input, autoencoder.layers[6].output)
        return encoder
    
    def get_decoder(self, weight): 
        autoencoder = load_model(weight)
        
        code_channel_length = int(self.opt.ae_type[len('code_length-'):len('code_length-')+1])
        print('code_channel_length', code_channel_length) ###
        
        input_decoder = Input(shape=(44, 26, code_channel_length))
        x = autoencoder.layers[-7](input_decoder) 
        for i in range(6,(2)-1,-1):
            x = autoencoder.layers[-i](x) 
        output_decoder = autoencoder.layers[-1](x) 
        decoder = Model(input_decoder, output_decoder)
        return decoder
    

#########################################################################################################
#########################################################################################################
class CompositeNN():
    def __init__(self, opt):
        self.opt = opt
        self.setup()

    def setup(self):
        self.n_val_stations = len(self.opt.val_stations.split('_')) 
        self.n_features = len(self.opt.features.split('_')) 
        self.domains = [x.split('_') for x in self.opt.domains.split('~')]
        
    def define_cnn(self, input_):
        output_ = Conv2D(filters=1, kernel_size=(3,3), 
                         activation='relu', padding='same', name='conv')(input_)
        return output_
    
    def define_cnn_test(self, input_):
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(4, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(4, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='relu')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        return decoded 
    
    def define_s2gMD(self, lats_lons):
        input_shape = (n_all_stations - self.n_val_stations, self.n_features)
        input_ = Input(shape=input_shape)
        
        grid_norm_MD = [self.get_gridSD(domain, k, weightKNN, input_) for (domain, k, weightKNN) in self.domains]
        grid_norm_concat = Concatenate(name='concat_grid')(grid_norm_MD)
        
        x = BatchNormalization(name='bn')(grid_norm_concat) 
        
        grid_hat = self.define_cnn_test(x) ###
        
        output_ = Lambda(self.select_from_grid, arguments={'lats_lons': lats_lons})(grid_hat)
        
        model = Model(input_, output_)
        
        for layer in model.layers: 
            if layer.name in ['s2c-air','c2g-air', 's2c-sat','c2g-sat']: layer.trainable = False ###
        
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'mse'])
        return model
    
    def get_s2gMD(self, weight):
        s2gMD_model = load_model(weight, custom_objects={'select_from_grid': self.select_from_grid})
        s2gMD_model_ = Model(s2gMD_model.input, s2gMD_model.layers[-2].output)
        return s2gMD_model_
    
    def select_from_grid(self, input_arr, lats_lons):
        epas=[input_arr[:, lat, lon, :] for lat,lon in lats_lons]
        output_arr=K.concatenate(epas)
        return output_arr
    
    def get_gridSD(self, domain, k, weightKNN, input_): 
        optionS2GSD = options.OptionS2GSD(
            domain=domain, k=k, weightKNN=weightKNN, 
            features=self.opt.features, val_stations=self.opt.val_stations, 
            ae_type=self.opt.ae_type, dnn_type=self.opt.dnn_type)
        
        modelS2GSD = station2gridSD_model.ModelS2GSD(optionS2GSD)
        s2c_model = modelS2GSD.s2c_model 
        c2g_model = modelS2GSD.c2g_model
        group = '_'.join([domain, k, weightKNN])
        s2c_model.name = 's2c-%s'%(domain) ### 's2c-%s'%(group)
        c2g_model.name = 'c2g-%s'%(domain) ### 'c2g-%s'%(group)
        
        code = s2c_model(input_)
        code_reshape = Reshape(target_shape=(44, 26, 4))(code) 
        grid_norm = c2g_model(code_reshape)
        return grid_norm 