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
For both models, you can train model just using below code.
```
python main.py
```

### Validate
If you want to validate models using saved model, follow below.

- $MODEL_NAME_1
```
python validate.py --model $MODEL_NAME --fold $FOLD_NUM --sigma $SIGMA_VALUE
```
If you follow this implementation using the model we provide, 
```
python validate.py --model model_main_fold_0_s_0.1_SF_TSP.pt --fold 0 --sigma 0.1
```
and you can get validation score of *f1: 0.8132, precision: 0.8023, recall: 0.8245


- $MODEL_NAME_2
```

```
