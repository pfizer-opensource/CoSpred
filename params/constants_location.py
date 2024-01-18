BASE_PATH = "./"
DATA_DIR = BASE_PATH + 'data/example/'
FILE_NAME = 'example'
TEMP_DIR = BASE_PATH + "temp/"
TRAINFILE = DATA_DIR + FILE_NAME + '_train.mgf'
TESTFILE = DATA_DIR + FILE_NAME + '_test.mgf'

# preprocess
PSM_PATH = DATA_DIR + FILE_NAME + "_PSMs.txt"
MGF_PATH = DATA_DIR + FILE_NAME + ".mgf"
MZML_PATH = DATA_DIR + FILE_NAME + ".mzML"

REFORMAT_USITITLE_PATH = DATA_DIR + FILE_NAME + ".mgf"
REFORMAT_TRAIN_USITITLE_PATH = DATA_DIR + "train_usi.mgf"
REFORMAT_TEST_USITITLE_PATH = DATA_DIR + "test_usi.mgf"
REFORMAT_TRAIN_PATH = DATA_DIR + "train_reformatted.mgf"
REFORMAT_TEST_PATH = DATA_DIR + "test_reformatted.mgf"
TRAINCSV_PATH = DATA_DIR + "peptidelist_train.csv"
TESTCSV_PATH = DATA_DIR + "peptidelist_test.csv"
TESTPEPTIDES_PATH = DATA_DIR + "test_peptides.csv"
TRAINDATA_PATH = DATA_DIR + "train.hdf5"
TESTDATA_PATH = DATA_DIR + "test.hdf5"
TRAINDATASET_PATH = DATA_DIR + "train_arrow/"
TESTDATASET_PATH = DATA_DIR + "test_arrow/"

# model
MODEL_SPECTRA = BASE_PATH + "model_spectra/"
MODEL_IRT = BASE_PATH + "model_irt/"
MODEL_DIR = MODEL_SPECTRA
MODEL_DIR_PROSIT = MODEL_SPECTRA + 'prosit/'
MODEL_DIR_TRANSFORMER = MODEL_SPECTRA + 'transformer/'

# examples
EXAMPLE_DIR = BASE_PATH + "examples/"

# prediction
PREDICT_FORMAT = 'msp'
PREDICT_INPUT = TESTCSV_PATH
REFERENCE_SPECTRA = REFORMAT_TEST_PATH
PREDICT_DIR = BASE_PATH + "prediction/"
PLOT_DIR = PREDICT_DIR + "plot/"
