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
crafting ../docker_run.yaml
```

### Start train 
```bash
train.sh -fswp_train.yml
```
### Test on data
```bash
run.sh -fswp_run.yml
```
### Test on simulator 
