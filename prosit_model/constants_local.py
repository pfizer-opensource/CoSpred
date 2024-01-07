BASE_PATH = "/Users/xuel12/Documents/Projects/seq2spec/CoSpred/"
DATA_DIR = BASE_PATH + "data/heladigest/"
DATA_PATH = BASE_PATH + "data/train.hdf5"
FILE_NAME = '2023_10_02_HeLa_200ng_03'
TEMP_DIR = BASE_PATH + "temp/"
TRAINFILE = DATA_DIR + FILE_NAME + '_train.mgf'
TESTFILE = DATA_DIR + FILE_NAME + '_test.mgf'

# preprocess
PSM_PATH = DATA_DIR + FILE_NAME + "_PSMs.txt"
MGF_PATH = DATA_DIR + FILE_NAME + ".mgf"
MZML_PATH = DATA_DIR + FILE_NAME + ".mzML"
REFORMAT_TRAIN_PATH = DATA_DIR  + "train_reformatted.mgf"
REFORMAT_TEST_PATH = DATA_DIR  + "test_reformatted.mgf"
TRAINCSV_PATH = DATA_DIR + "train.csv"
TESTCSV_PATH = DATA_DIR + "test.csv"
TRAINDATA_PATH = DATA_DIR + "train.hdf5"
TESTDATA_PATH = DATA_DIR + "test.hdf5"

# model
MODEL_SPECTRA = BASE_PATH + "local_training_tf2/model_spectra"
MODEL_IRT = BASE_PATH + "local_training_tf2/model_irt/"
MODEL_DIR = MODEL_SPECTRA

# examples
EXAMPLE_DIR = BASE_PATH + "examples/"

# prediction
PREDICT_FORMAT = 'msp'
# PREDICT_INPUT = EXAMPLE_DIR + "peptidelist.csv"
# REFERENCE_SPECTRA = DATA_DIR + 'test_reformatted.mgf'
PREDICT_INPUT = TESTCSV_PATH
REFERENCE_SPECTRA = REFORMAT_TEST_PATH
PREDICT_DIR = BASE_PATH + "prediction/"
PLOT_DIR = PREDICT_DIR + "plot/"

