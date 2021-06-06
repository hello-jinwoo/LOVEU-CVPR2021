FILE_LIST = 'file_list_5fold.pkl'
ANNOTATION_PATH = 'k400_all.pkl'
TEST_ANNOTATION_PATH = 'test_len.json'
DATA_PATH = '../data/LOVEU_both_hr_padded/train_val/'
DATA_PATH_2 = '../data/LOVEU_both_hr/train_val/'
TEST_DATA_PATH = '../data/LOVEU_both_hr_padded/test/'
PRED_PATH = 'k400_pred.pkl' 
MODEL_SAVE_PATH = './models/'

DEVICE = 'cuda'

FEATURE_DIM = 2304 + 4096 # 6400 for SF_TSN, 6912 for SF_TSN_TSP
FEATURE_DIM_1 = 2304 # 2304 for both SF_TSN, and SF_TSN_TSP
FEATURE_DIM_2 = 4096 # 4096 for SF_TSN, 4608 for SF_TSN_TSP

FEATURE_LEN = 40 # DO NOT CHANGE
TIME_UNIT = 0.25 # DO NOT CHANGE

ENCODER_HIDDEN = 1024
DECODER_HIDDEN = 128
AUX_LOSS_COEF = 0.5

BATCH_SIZE = 256
LEARNING_RATE = 1e-4
DROP_RATE = 0.2

GLUE_PROB = 0.3 # Probability of glueing augmentation 
INTERPOLATION_PROB = 0.2 # Probability of data of DATA_PATH_2

VAL_VIDEOS = ["hlm7ShpS1z0", "jw1kPjGu1YE", "u959kyRLWdQ", "z1-T4zHXvSY", "z1-U0uYFY3Q"] # files in visualizing folder

THRESHOLD = 0.1 # Minimum score to be event boundary
SIGMA_LIST = [-1, 0.4] # List of sigma values of gaussian filtering in validation
TEST_THRESHOLD = 0.808
GOAL_SCORE = 0.815 # Train ends when validation score gets here

PATIENCE = 10 # Patience for early stopping
