# LOVEU-CVPR2021

This is the code for track 1 of competition LOVEU@CVPR2021.

We have two main models. One of them is *main (using_similarity_map)* using similarity map, and the other one is *simple(sf_tsn_each_branch)* using comparably simple networks.(transformer, RNN, and CNN)

We provide the simple way to implement(train, validate, test using ensemble) our code below :)

## Setup
Here, we provide our basic setup. 
- python 3.8
- torch 1.8.1
- numpy 1.19.2
- matplotlib 3.4.1
- tqdm 

## Video Features
<!-- You can downlaod our video feuatures [here](https://drive.google.com/drive/folders/1AJl177kLvl1YtaFBb9QmiUAQ5o5qsjq9?usp=sharing). -->

You may locate feature data in 'data' folder of this repository.

## Materials 
You can download materials [here](https://drive.google.com/drive/folders/1Z5rFZlRIjBNDcys-Y1we9nXOB1Al2x3y?usp=sharing).

You may locate *file_list_5fold.pkl* in each model's folder, *model files (.pt)* in folder 'models' of each model folder, and *prob_result files (.pkl)* in folder 'prob_results' of ensemble folder.

## Implementation
You can change some values in config.py in both models. 

### Train
For both models, you can train model just using below code.
```
python main.py
```

### Validate
If you want to validate models using saved model, follow below.

- *main*
```
python validate.py --model $MODEL_NAME --fold $FOLD_NUM --sigma $SIGMA_VALUE
```
If you follow this implementation using the model we provide, 
```
python validate.py --model models/model_main_fold_0_s_0.1_SF_TSP.pt --fold 0 --sigma 0.1
```
and you can get validation score of *f1: 0.8132, precision: 0.8023, recall: 0.8245*

<hr>

- *simple*
```
python validate.py --model_sf $MODEL_SF_NAME --model_tsn $MODEL_TSN_NAME --sigma $SIGMA_VALUE --fold $FOLD_NUM
```
If you follow this implementation using the model we provide, 
```
python validate.py --model_sf models/model_sf_fold_4_s_-1_SF_TSP.pt --model_tsn models/model_tsn_kim_fold_4_s_-1_SF_TSP.pt --sigma -1 --fold 4
```
and you can get validation score of *f1: 0.8119, precision: 0.7921, recall: 0.8327*


### Test with ensemble
We predict the result by ensembling models from different folds(0~4) and model architecture(*main* and *simple*).

We save a probability score for each model and use it to produce final prediction.

With the probability scores, you can predict the final event boundary following below code in ensemble folder.
```
python test.py --ver $VERSION_NAME_YOU_WANT
```

Then, there will be the test result in results folder.


