BASE_PATH = "/mnt/data/CoSpred/"
# DATA_PATH = BASE_PATH + "data/train.hdf5"
# DATA_DIR = BASE_PATH + "data/propel2017_byion/"
# FILE_NAME = 'propel2017_massiveKBv1_200kpep'
# DATA_DIR = BASE_PATH + "data/propel2017/"
# FILE_NAME = 'propel2017_massiveKBv1_200kpep'
# DATA_DIR = BASE_PATH + 'data/massiveKBv2synthetic/'
# FILE_NAME = 'massiveKBv2synthetic_889kpep'
# DATA_DIR = BASE_PATH + 'data/phospho/'
# FILE_NAME = 'phospho_sample1'
# DATA_DIR = BASE_PATH + 'data/phospho_forBYion/'
# FILE_NAME = 'phospho_sample1'
# DATA_DIR = BASE_PATH + 'data/heladigest/'
# FILE_NAME = '2023_10_02_HeLa_200ng_03'
DATA_DIR = BASE_PATH + 'data/PXD014525/'
FILE_NAME = '20180810_QE3_nLC3_AH_DDA_Yonly_ind_03'
TEMP_DIR = BASE_PATH + "temp/"
TRAINFILE = DATA_DIR + FILE_NAME + '_train.mgf'
TESTFILE = DATA_DIR + FILE_NAME + '_test.mgf'

# preprocess
PSM_PATH = DATA_DIR + FILE_NAME + "_PSMs.txt"
MGF_PATH = DATA_DIR + FILE_NAME + ".mgf"
MZML_PATH = DATA_DIR + FILE_NAME + ".mzML"

REFORMAT_USITITLE_PATH = DATA_DIR + FILE_NAME + ".mgf"
REFORMAT_TRAIN_USITITLE_PATH = DATA_DIR  + "train_usi.mgf"
REFORMAT_TEST_USITITLE_PATH = DATA_DIR  + "test_usi.mgf"
REFORMAT_TRAIN_PATH = DATA_DIR  + "train_reformatted.mgf"
REFORMAT_TEST_PATH = DATA_DIR  + "test_reformatted.mgf"
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
# MODEL_SPECTRA = BASE_PATH + "prosit_model/model_spectra/"
MODEL_DIR = MODEL_SPECTRA
MODEL_DIR_PROSIT = MODEL_SPECTRA + 'prosit/'
MODEL_DIR_TRANSFORMER = MODEL_SPECTRA + 'transformer/'

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

#DATA_PATH = "/root/data.hdf5"
#MODEL_SPECTRA = "/root/model_spectra/"
#MODEL_IRT = "/root/model_irt/"
#OUT_DIR = "/root/prediction/"

