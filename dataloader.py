import pandas as pd
import numpy as np
import math
from collections import defaultdict
import json
import sys

def load_dataframe(path):
    return pd.read_csv(path)

def load_user_data(usr_df, label, embedding_type=None, time_step='week'):
    """
    args:
        usr_df: DataFrame containing each user's weekly/daily data per row
        label: String for which field to treat as label
        embedding_type: String to toggle between 'w2v' or 'bert'. Default is univariate forecast
        time_step: String to toggle between week or daily data. Using the column name(week or day_id)
    return:
        dictionary of key = user_id and value = sequence of user data over all available time
    """
    usr_dict = defaultdict(list)

    for idx,row in usr_df.iterrows(): # load all users and format into dictionary
        if embedding_type:
            embedding = np.array(json.loads(row[embedding_type])) # embeddings are stored as json
        

        target = row[label]
        
        try:
            other_vars = embedding
        except:
            other_vars = []

        # append other variables here if desired (i.e. intensity along with embeddings for Aff pred)
        # if label = 'affect':
        #   other_vars.append(row['intensity'])

        if time_step == 'week':
            key = 'week'
        else:
            key = 'day_id'
                
        if len(other_vars) > 0:
            usr_data = (row[key], np.append(target, other_vars))
        else:
            usr_data = (row[key], target)
            
        usr_dict[row['user_id']].append(usr_data)
    
    return usr_dict

def gen_train_data(user_data, n):
    """
    args:
        user_data: Dictionary containing each user's full history as a sequence
        n: Integer denoting the maximum history to use for the model
    return:
        train_data: numpy array of chunked user history
        train_labels: numpy array of label per user history sequence 
    """

    train_data, labels = [], []
    for k,v in user_data.items():
        usr_all_history = user_data[k][:15] # eahc user has maximum 14 time-steps
        usr_train_data = []
        usr_train_labels = []

        for i in range(15-n): # only go back as far as n
            curr_train = []
            curr_label = []
            for j in range(n): # for each time-step
                if j < n - 1: 
                    curr_train.append(usr_all_history[j+i][1])
                elif j == n -1:
                    curr_train.append(usr_all_history[j+i][1])

                    # assumes multi-variate, catches univariate case
                    # [0] grabs the target which is always first element of that week
                    try:
                        curr_label.append(usr_all_history[j+i+1][1][0])              
                        features = [f for week in curr_train for f in week]
                    except:
                        curr_label.append(usr_all_history[j+i+1][1])
                        features = [f for f in curr_train]
            
            train_data.append(features)
            labels.append(curr_label[0])

    return np.array(train_data), np.array(labels)

def gen_test_data(user_data, n):
    """
    args:
        user_data: Dictionary containing each user's full history as a sequence
        n: Integer denoting the maximum history to use for the model
    return:
        test_data: numpy array of chunked user history
        test_labels: numpy array of label per user history sequence 
    """
    test_data, test_labels = [], []
    for k,v in user_data.items():
        usr_test_history = user_data[k][-5:] # Grab remaining weeks in user's sequence for testing
        usr_test_data = []
        usr_test_labels = []
        for i in range(4): # 4 test weeks
            features = []
            for j in range(1,n):
                features = np.append(user_data[k][:][(-5+i)-j][1], features)

            usr_test_embeds = np.append(features, usr_test_history[i][1])
            test_data.append(usr_test_embeds)

            try:
                test_labels.append(usr_test_history[i+1][1][0])
            except:
                test_labels.append(usr_test_history[i+1][1])
            
    return test_data, test_labels

if __name__  == '__main__':
    print('Testing data loader')

    df = load_dataframe('./data/weekly_all_labels.csv')
    usr_seqs = load_user_data(df, 'affect', embedding_type='w2v')
    

    train_data, train_labels = gen_train_data(usr_seqs, 5)
    print(train_labels[:5])
    test_data, test_labels = gen_test_data(usr_seqs,5)
    print(test_labels[:5])
