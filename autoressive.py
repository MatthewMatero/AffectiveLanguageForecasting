import math
import numpy as np
from os import listdir
import pandas as pd
import json
from sklearn.linear_model import Ridge
import argparse
import copy
from scipy.stats import sem
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import sys

from dataloader import load_dataframe, load_user_data, gen_train_data, gen_test_data


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Set AR params")
    parser.add_argument('-n', type=int, action="store", dest="n", help='number of time-steps to use as history')
    parser.add_argument('--embedding_type', action="store", dest="embedding_type", help='toggle between w2v and bert embeddings')
    args = parser.parse_args()

    df = load_dataframe('./data/weekly_all_labels.csv')
    usr_seqs = load_user_data(df, 'affect', embedding_type=args.embedding_type)
    train_data, train_labels = gen_train_data(usr_seqs, args.n)
    test_data, test_labels = gen_test_data(usr_seqs, args.n)
    

    # build train df
    train_df = pd.DataFrame(train_data)
    train_labels_df = pd.DataFrame(train_labels)
    train_labels_df.columns = ['label']
    train_labels_df['label'] = train_labels_df['label'].astype(float)

    # build test df
    test_df = pd.DataFrame(test_data).dropna()
    model = Ridge(alpha=.01, normalize=False)

    model.fit(train_df, train_labels_df.label)

    test_preds = model.predict(test_df)
    test_mse = mean_squared_error(test_labels, test_preds)
    test_corr = pearsonr(test_labels, test_preds)
    
    print('#########RESULTS##########')
    print('OOS Time User MSE: ', test_mse)
    print('OOS Time User Corr: ', test_corr)
    
    # don't test against missing data for daily data! 
    #if args.daily_data:
    #    drop_indices = [i for i,x in enumerate(test_labels) if x == 0]
    #    test_labels = [test_labels[i] for i,x in enumerate(test_labels) if i not in drop_indices ]
    #    test_preds = [test_preds[i] for i,x in enumerate(test_preds) if i not in drop_indices ]
