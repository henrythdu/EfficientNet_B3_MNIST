# MNIST Digit Classification with EfficientNetB3 in PyTorch

## Introduction
This repository contains a PyTorch implementation of the EfficientNetB3 model for classifying handwritten digits from the MNIST dataset. The project demonstrates how transfer learning can be applied to a smaller, well-defined problem. The goal is to use a small model to achieve high accuracy.

## Dataset
The MNIST dataset is a large database of handwritten digits commonly used for training various image processing systems. It contains 60,000 training images and 10,000 testing images. More details can be found here.

## Model
EfficientNetB3 is part of the EfficientNet family, which represents a scaling method that uniformly scales all dimensions of depth/width/resolution using a compound coefficient. 


## Requirements
```
torch==2.1.2
torchvision==0.16.2
tqdm==4.66.1
matplotlib==3.7.5
numpy==1.26.4
pandas==2.2.0
sklearn==1.2.2
torchinfo==1.8.0
```

## Usage
1. Clone the repository: 
```
git clone https://github.com/henrythdu/Food-Vision-Big.git
```

2. Install the dependencies:

```python
pip install -r requirements.txt
```

3. Run the training notebook:
```
FoodVisionBig_training.ipynb
```

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgments
- Food101 dataset creators
- Original authors of the EfficientNet model