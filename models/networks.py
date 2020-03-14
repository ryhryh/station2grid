import os
import pandas as pd
from keras.layers import Input, Flatten, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation 
from keras.models import Model, load_model
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import Sequence

class AE():
    def __init__(self, opt): 
        self.opt = opt
        
    def define_ae(self):
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

        # autoencoder
        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        return autoencoder
        
    def get_encoder(self, weight): 
        autoencoder_best = load_model(weight)
        encoder_best = Model(autoencoder_best.input, autoencoder_best.layers[6].output)
        return encoder_best
    
    def get_decoder(self, weight): 
        autoencoder_best = load_model(weight)
        
        input_decoder = Input(shape=(44, 26, 4))
        x = autoencoder_best.layers[-7](input_decoder) 
        for i in range(6,(2)-1,-1):
            x = autoencoder_best.layers[-i](x) 
        output_decoder = autoencoder_best.layers[-1](x) 
        decoder_best = Model(input_decoder, output_decoder)
        return decoder_best
    
    def get_callbacks(self, weight_dir):
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=3, min_delta=0.01)
        
        file_name = "%s-epoch_{epoch:02d}-val_loss_{val_loss:.3f}.hdf5"%(self.opt.model_name)
        path = os.path.join(weight_dir, file_name)
        checkpointer = ModelCheckpoint(filepath=path, verbose=0, period=1,monitor='val_loss', save_best_only=True, mode='min')
        callbacks = [early_stopping, checkpointer]
        return callbacks
        

class FCNN():
    def __init__(self, opt):
        self.opt = opt
    
    def define_fcnn(self, n_val_stations, n_features, codeLength):
        num_station = 73 - n_val_stations
        num_encoded = codeLength # 4576
        input_ = Input(shape=(num_station, n_features))
        x = Flatten()(input_)
        x = Dense(300, activation='relu')(x)
        output_ = Dense(num_encoded, activation='linear')(x)
        model = Model(input_, output_)
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'mse'])
        return model
    
    def get_fcnn(self, weight):
        nn_best = load_model(weight)
        return nn_best
    
    def get_callbacks(self, weight_dir):
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=3, min_delta=0.01)
        
        file_name = "%s-epoch_{epoch:02d}-val_loss_{val_loss:.3f}.hdf5"%(self.opt.model_name)
        path = os.path.join(weight_dir, file_name)
        checkpointer = ModelCheckpoint(filepath=path, verbose=0, period=1,monitor='val_loss', save_best_only=True, mode='min')
        callbacks = [early_stopping, checkpointer]
        return callbacks
    