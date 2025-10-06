# PhotoDocsCreator

## Description

This project is an open source photo processing tool, the output of which is passport photos.

## Example

![An example of how PhotoDocsCreator works](src/tests/ResultExample.jpg)

## Photo requirements

To get a well-processed photo, you need the original photo to meet the following requirements:

* The image must be taken in good resolution (minimum full HD for high dpi).
* The light in the photo should be natural and illuminate the person evenly.
* The background of the entrance photo should be adequate.
* A person's head should look straight like his gaze, and keep his hands along his torso. The facial expression should be neutral.
* There should be no headdresses (if the person's religion allows it). Hair should not cover the face.
* Only eyeglasses are allowed, and only if a person is constantly wearing them. Moreover, in this case, the eyes should be clearly visible, there should be no glare.

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