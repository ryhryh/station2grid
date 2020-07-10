# station2grid

A Keras implementations for station2grid: Grid Generation via Domain Transfer.

The [slide](https://docs.google.com/presentation/d/1Spu1Stuj2Zqy9uz5kf2446bstAXVhWNGOJZnA7yjiGw/edit?usp=sharing) includes a more detailed description.


## Introduction
The goal is to input a fixed amount of real-time EPA air quality station data at a specific time, and then output 
a high-resolution PM2.5 forecast of 1km x 1km in Taiwan. We use the concept of transfer learning to transfer the knowledge 
learned from multiple source domains towards the target domain.

![alt text](gif/cyclegan.png)

## Model Architecture
Single-Domain Transfer Architecture | Multi-Domains Transfer Architecture
:-------------------------:|:-------------------------:
![alt text](gif/cycle-s-m.gif)  |  ![alt text](gif/cycle-m-s.gif)

## Results

## Prerequisites
- Python 3.6
- Keras 2.2.4

## Usage
- Clone the repository
- Download datasets
- Train
- Test
