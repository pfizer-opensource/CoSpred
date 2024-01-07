import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data_dir = "sdb/TUMData/data/PXD000612/"
inputfile = data_dir + "msms.txt"
rawfiles = data_dir + "phosphorawfile.txt"
outputfile = data_dir + "PXD000612msms.txt"

# To get csv file for each rawfiles.
#df = pd.read_csv(inputfile, sep='\t', lineterminator='\n')
#dfs = dict(tuple(df.groupby('Raw file')))
#for key, value in dfs.items():
#    value.to_csv(data_dir + str(key) + ".csv", index=False)

# To get subset of phospho data

df1 = pd.read_csv(inputfile, sep='\t')
#df2 = pd.read_csv(rawfiles, sep='\t')

#result = pd.merge(df2, df1, how='inner', on='Raw file')
#result.to_csv(outputfile, sep='\t', index=False)

## To get batches
#df1 = pd.read_csv(inputfile, sep='\t')
#list_df = np.array_split(df1, 2)
#for i, inf in enumerate(list_df):
#    inf.to_csv(data_dir + str(i) + ".csv", index=False)

## To split the dataset
X_train, X_test = train_test_split(df1, test_size=0.5, random_state=42)
X_train.to_csv(data_dir + "first" + ".csv", index=False)
print(X_train.shape)
X_test.to_csv(data_dir + "second" + ".csv", index=False)
print(X_test.shape)
