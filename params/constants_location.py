BASE_PATH = "./"
DATA_DIR = BASE_PATH + 'data/example/'
FILE_NAME = 'example'
TEMP_DIR = BASE_PATH + "temp/"
TRAINFILE = DATA_DIR + FILE_NAME + '_train.mgf'
TESTFILE = DATA_DIR + FILE_NAME + '_test.mgf'

# preprocess
PSM_PATH = DATA_DIR + FILE_NAME + "_PSMs.txt"
MAPPINGFILE_PATH = DATA_DIR + FILE_NAME + "_InputFiles.txt"
MGF_PATH = DATA_DIR + FILE_NAME + ".mgf"
MZML_PATH = DATA_DIR + FILE_NAME + ".mzML"
MGF_DIR = DATA_DIR + "mgf/"
MZML_DIR = DATA_DIR + "mzml/"


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
PREDICT_ORIGINAL = DATA_DIR + "peptidelist_test.csv"
PREDICTCSV_PATH = DATA_DIR + "peptidelist_predict.csv"
PREDDATA_PATH = DATA_DIR + "predict.hdf5"
PREDDATASET_PATH = DATA_DIR + "predict_arrow/"
PREDICT_DIR = BASE_PATH + "prediction/"
PREDICT_HDF5_DIR = PREDICT_DIR + "hdf5/"
PREDICT_BATCH_DIR = PREDICT_DIR + "prediction_batches/"
PREDICT_BATCH_RESULT_FILE = PREDICT_BATCH_DIR + "prediction_batch_combined.h5"
PREDICT_BATCH_COMBINED_FILE = PREDICT_BATCH_DIR + "combined_batch_result.h5"
PREDICT_CHUNK_DIR = PREDICT_DIR + "prediction_chunks/"
PREDICT_CHUNK_RESULT_FILE = PREDICT_CHUNK_DIR + "prediction_chunk_combined.h5"
PREDICT_RESULT_FILE = PREDICT_DIR + "spectrum_prediction_result.h5"
PREDICT_LIB_DIR = PREDICT_DIR + "prediction_library/"
PREDICT_LIB_FILENAME = "speclib_prediction"

# plot
PLOT_DIR = PREDICT_DIR + "plot/"

# logging
LOGS_DIR = BASE_PATH + 'logs/'
