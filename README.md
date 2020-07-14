# station2grid

A Keras implementation of station2grid: Grid Generation via Domain Transfer. <br>
The [slide](https://docs.google.com/presentation/d/1Spu1Stuj2Zqy9uz5kf2446bstAXVhWNGOJZnA7yjiGw/edit?usp=sharing) includes a more detailed description.


## Introduction
The goal is to input historical data from several air quality stations, and output air quality of 1km*1km fine-grained map in Taiwan. We use the concept of transfer learning to transfer the knowledge learned from multiple source domains towards the target domain.

<img src="imgs/transfer_learning" width="250">


## Model Architecture
The model can be divided into two parts.

1) Single-Domain Transfer: <br>
The Single-Domain Transfer network composed of Convolutional Autoencoders and Fully Connected Networks aims to learn the spatial correlation between fine-grained grids from one source domain.

2) Multi-Domains Transfer: <br>
The Multi-Domains Transfer network composed of multiple pre-trained models from source domains aims to combine those models via a U-net like Convolutional network to output a fine-grained map of PM2.5.  

Single-Domain Transfer | Multi-Domains Transfer
:-------------------------:|:-------------------------:
<img src="imgs/single_domain_transfer" width="400">  |  <img src="imgs/multi_domains_transfer" width="400">

## Experiments
### Datasets
<img src="imgs/dataset" width="500">

### Results <br>
- The SD-MS model reach 2nd best performance in urban areas, and the SD-SAT model 2nd best performance in mountainous areas.
- The MD model performs best in both urban areas and mountainous areas.

<img src="imgs/city_" width="500">
<img src="imgs/mountain_" width="500">

## Prerequisites
- Python 3.6
- Keras 2.2.4

## Usage
### Clone the repository
```bash
$ git clone https://github.com/ryhryh/station2grid.git

```

### Download datasets
```bash
$ cd station2grid/datasets/npy
$ python ./download_data.py
```

### Train grid2code
train autoencoder to do dimension reduction.
- domain: air or sat
```bash
$ cd station2grid/experiments
$ python ./grid2code.py --domain sat
```

### Train / Test station2grid
input all stations exclude valid_station, output fine-grained map of Taiwan.
- isTrain: 0 or 1
- domain: air or sat
- feature: pm25, pm25_PM10, pm25_PM10_CO2 ... 
- valid_station: Tainan, Linkou, ...
```bash
$ cd station2grid/experiments
$ python ./station2grid.py --isTrain 1 --feature pm25 --valid_station Tainan --domain sat
$ python ./station2grid.py --isTrain 0 --feature pm25 --valid_station Tainan --domain sat

```
