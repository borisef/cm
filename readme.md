# cm
######  by borisef

## About 

Implements color classifier training/testing/freezing in python & tensorflow & keras


## Installation

- Clone the project:
```sh
   git clone https://github.com/borisef/cm.git
```

## Data preparation 

Most simple and intuitive data: rgb images in folders named as names of classes: "red", "white", "green" , etc.
Supports folder-in-folder structure (for example for augmented images) 

## How To Run It

Entry points are those two scripts: ```ColorNet_TestGeneral.py```, ```ColorNet_TrainGeneral.py```
The output folder after train and test will contain:
  *  saved models -  of several types (we used the on in folder `k2tf_dir`)
  *  statistics - (confusion matrices, histograms etc) 
  *  checkpoints along training 
