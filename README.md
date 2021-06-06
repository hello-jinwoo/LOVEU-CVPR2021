# LOVEU-CVPR2021

This is the code for track 1.1 and track 1.2 of competition LOVEU@CVPR2021.

We have two main models. One of them is $MODEL_NAME_1$ using similarity map, and the other one is $MODEL_NAME_2$ using comparably simple networks.(transformer, RNN, and CNN)

We provide the simple way to implement(train, validate, test using ensemble) our code below and if you have any questions, feel free to email me <jinwoo-kim@yonsei.ac.kr> :)

## Setup
Here, we provide our basic setup. 
- python 3.8
- torch 1.8.1
- numpy 1.19.2
- matplotlib 3.4.1
- tqdm 

## Video Features
You can downlaod our video feuatures [here]().

You may locate feature data in 'data' folder of this repository.

## Materials 
You can download materials [here]().

You may locate file_list_5fold.pkl in each model's folder, and the model files in folder 'models' of each model.

## Implementation
You can change some values in config.py in both models. 

### Train
For both 
