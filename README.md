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
├── utils
└── docker 
```

##### 2. Download the data
Download convlstm training data, and data unet training. Put the received datasets in the **data** folder.

##### 3. Download models
Download the unet model and put it in a folder
*checkpoints/unet*

Download the convlstm model and put it in a folder
*checkpoints/convlstm*

##### 4. Download requirements.txt
```bash
pip install -r requirements.txt
```

## Docker 
We use [crafting](https://pypi.org/project/crafting/) to automate our experiments. 
You can find an example of running such a pipeline in ```run.yaml``` file. 
You need to have installed Docker, Nvidia drivers, and crafting package. 

The crafting package is available in PyPI:
```bash
pip install crafting
```


To build the image run the command below in ```docker``` folder:
```bash
sh build.sh
```

To run an experiment specify target command in ```command``` field in ```run.yaml``` file and call crafting:
```bash
crafting configs/docker_run.yaml
```

## Start train 
```bash
train.sh -fswp_train.yml
```
## RUN
### Test on data
```bash
python3 utils/predict_on_tdataset.py ./configs/run_solar_fswp.yaml
```
### Test on simulator 
```bash
python3 utils/run_on_env.py configs/run_star_fswp.yaml False checkpoints/convlstm/CONV_LSTM_run_EricWright_94ed9438-8fae-4fc8-8afa-85409d0c6f46.pth  
```
