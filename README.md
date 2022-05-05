# FSWP - fast wave-front prediction
____
## Install

##### 1. Project structure
First of all, make sure the project structure matches the one below:

```
├── fswp_train.yml
├── fswp_run.yml
├── _data
│   └── # pickle files for model training
├── checkpoints
│   └── # .pth models 
├── models
│   ├── unet #unet trainer
│   └── convlstm #convlstm trainer
└── utils
```

##### 2. Download the data
Download convlstm training data, and data unet training. Put the received datasets in the **data** folder.

##### 3. Download models
Download the unet model and put it in a folder
*checkpoints/unet*

Download the convlstm model and put it in a folder
*checkpoints/convlstm*

### Start train 

> train.sh -fswp_train.yml
### Test on data

> run.sh -fswp_run.yml
### Test on simulator 
