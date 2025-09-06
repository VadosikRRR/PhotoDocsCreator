# PhotoDocsCreator

## Description

This project is an open source photo processing tool, the output of which is passport photos.

## Example

![An example of how PhotoDocsCreator works](src/tests/ResultExample.jpg)

## How to install and run (On Linux)

#### Step 1: Clone the repository

``` bash
git clone git@github.com:VadosikRRR/PhotoDocsCreator.git
cd PhotoDocsCreator
```

#### Step 2: Install packages

``` bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

#### Step 3: Download weights

Download the weights from the [link](https://drive.google.com/file/d/14-hSmX8IE1_WIX26WR-hSXdyPXb1wvTy/view?usp=sharing)

Move the weight file to the folder: PhotoDocsCreator/src/PhotoDocsCreator/face_parsing/res/cp
#### Step 4: Launch project

``` bash
cd src/api
python main.py
```

## Link to face-parsing

The model from the [zllrunning repository](https://github.com/zllrunning/face-parsing.PyTorch) was used in the work

## Link to BiRefNet

The model from the [ZhengPeng7 repository](https://github.com/ZhengPeng7/BiRefNet) was used in the work