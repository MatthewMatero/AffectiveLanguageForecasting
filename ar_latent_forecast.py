db = 'twitterGender'
pswd = ''
usr = 'mmatero'
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
from sklearn.metrics import mean_squared_error
from pprint import pprint
import math
import numpy as np
from os import listdir
import pandas as pd
import json
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
import argparse
from plotter import my_plotter
import matplotlib.pyplot as plt
import pickle
import copy
from scipy.stats import sem
from scipy.stats import pearsonr
import sys

# Used to save user preds to plot with other data
def save_obj(obj, name):
    with open(name + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

parser = argparse.ArgumentParser(description="Set AR params")
parser.add_argument('-w', action="store", dest="w")
parser.add_argument('--uni', action="store_true", dest="univariate")
parser.set_defaults(univariate=False)
parser.add_argument('--w2v', action="store_true", dest="word_vec")
parser.set_defaults(word_vec=False)
parser.add_argument('--svr', action="store_true", dest="svr")
parser.set_defaults(svr=False)
parser.add_argument('--gbr', action="store_true", dest="gbr")
parser.set_defaults(gbr=False)
parser.add_argument('--intensity', action="store_true", dest="ints")
parser.add_argument('--bert', action="store_true", dest="bert")
parser.add_argument('--4layer', action="store_true", dest="four_layer")
parser.add_argument('--tempor', action="store_true", dest="temp") 
parser.add_argument('--plang', action="store_true", dest="pure_lang")
parser.add_argument('--8emo', action="store_true", dest="emo")
parser.add_argument('--liwc', action="store_true", dest="liwc")
parser.add_argument('-target', action="store", dest="target")
parser.add_argument('-k_folds', action="store", dest="k_folds", type=int)
parser.add_argument('--30k', action="store_true", dest="full_set")
parser.add_argument('--daily', action="store_true", dest="daily_data")

args = parser.parse_args()
window = int(args.w)
uni = False
model_name = str(window)
if args.univariate:
    print('Calculating univariate model')
    uni = True
    model_name += '_uni'

lang = False
if args.word_vec or args.bert:
    uni = False
    print('Calculating lang based models')
    lang = True
    model_name += '_w2v'

svr = False
if args.svr:
    model_name += '_svr'
    svr = True

gbr = False
if args.gbr:
    model_name += '_gbr'
    gbr = True

intense = False
if args.ints:
    model_name += '_ints'
    intense = True

main_table = 'usr2k_3twtperwk_full'

if args.full_set:
    main_table = 'usr30k_2kfilter'
    #main_table = 'usr30k_3twtperwk_diff'

if args.daily_data:
    main_table = 'usr2k_daily_diff'

if args.temp:
    main_table = 'usr2k_3twtperwk_tempor_full'
    print('Main Table: ', main_table)

if args.emo:
    main_table = 'usr2k_8emo_diff'
    print('Main Table: ', main_table)

if args.liwc:
    main_table = 'usr2k_liwc_diff'
    print('Main Table: ', main_table)

embed_table = 'usr30k_embeddings_diff'
if args.bert:
    if args.four_layer:
        embed_table = 'usr2k_bert4layer_nmf50_diff' 
    else:
        embed_table = 'usr2k_bert_nmf50_diff' 

    print('Embeds: ', embed_table)

myDB = URL(drivername='mysql', host='localhost',
    database='twitterGender',
    query={ 'read_default_file' : '~/.my.cnf' }
)

engine = create_engine(myDB)
conn = engine.connect() 

if args.bert or args.word_vec:
    sel = conn.execute("select a.*, b.embedding as emb from " + main_table + " a join " + embed_table + " b on a.user_id = b.user_id and a.week = b.week")
else:
    sel = conn.execute("select * from " + main_table)

usr_df = pd.DataFrame(sel.fetchall())
usr_df.columns = sel.keys()

undiff_sel = conn.execute("select * from usr_3twtsPerWk_preKNN")
undiff_df = pd.DataFrame(undiff_sel.fetchall())
undiff_df.columns = undiff_sel.keys()

def load_user_data():
    usr_dict = dict()
    for idx,row in usr_df.iterrows(): # load all users and format into dictionary
        #if row['user_id'] != 5728:
        #    continue
        
        if args.word_vec or args.bert:
            #try:
            embedding = np.array(json.loads(row['emb']))
            #except:
                #embedding = row['emb']

        if intense:
            bg = row['affect']
            target = row['intensity']
        elif args.temp:
            is_past = row['is_past']
            is_pres = row['is_pres']
            is_future = row['is_future']

            bg = [is_past, is_pres, is_future]
            target = row['ppf']
        elif args.emo:
            target = row[args.target]
        elif args.liwc:
            target = row[args.target]
        elif args.target == 'embed':
            target = embedding[0] # first bert ele
            embedding = embedding[1:]
        else: # default is affect forecast
            if row['msg_count'] < 0:
                bg = 0
                target = 0
            else:
                bg = row['intensity'] 
                target = row['affect']

        if uni:
            if args.daily_data:
                row_data = (row['day_id'], [target])    
            else:
                row_data = (row['week'], [target])
        elif lang and not args.pure_lang: 
            row_data = (row['week'],np.append(np.append(target, bg), embedding))
        elif lang and args.pure_lang: # doesn't use background vars
            row_data = (row['week'],np.append(target, embedding))
        else:
            row_data = (row['week'], np.append(target, bg))
                
        try:
            usr_dict[row['user_id']].append(row_data)
        except:
            usr_dict[row['user_id']] = [row_data]
    #print(usr_dict)
    return usr_dict

def gen_train_data_fixed(user_data,n):
    train_data, labels = [], []
    for k,v in user_data.iteritems():
        #print('User: ', k)
        usr_embeds = user_data[k][:15]
        usr_train_data = []
        usr_train_labels = []
        for i in range(15-n):
            curr_train = []
            curr_label = []
            for j in range(n):
                if j < n - 1: # just add difference data until we get to last week data
                    # all but mc_aff + langs
                    if uni:
                        curr_train.append(usr_embeds[j+i][1][:1])
                    elif lang:
                        curr_train.append(usr_embeds[j+i][1][:]) # all feats
                    else:
                        if args.temp:
                            curr_train.append(usr_embeds[j+1][1][:4]) # ppf and assoc dims
                        else:
                            curr_train.append(usr_embeds[j+i][1][:2]) 
                
                elif j == n - 1: # all avail data at T-1
                    curr_train.append(usr_embeds[j+i][1])
                    aff = usr_embeds[j+i+1][1][0] # next week(just aff)
                    curr_label.append(aff)

            features = [f for week in curr_train for f in week]
            train_data.append(features)
            labels.append(curr_label[0])
    
    #print(train_data[0]) 
    #print(len(train_data[0]))
    return np.array(train_data), np.array(labels)

def gen_test_data(user_data, w):
    test_data, test_labels = [],[]
    for k,v in user_data.iteritems():
        usr_embeds = user_data[k][-5:] # -5 for difference tests ([16,17],[17,18],[18,19],[19,20])
        usr_test_data = []
        usr_test_labels = []
        for i in range(4): # 4 for differences
            features = [] # get previous weeks features for testing
            for j in range(1,w): # start at 1 to avoid duplicating training week into test
                if lang: # if lang then give all features every week
                    features = np.append(user_data[k][:][(-5+i)-j][1][:], features)
                elif uni: # uni only gets target variable history
                    features = np.append(user_data[k][:][(-5+i)-j][1][:1], features)
                else:
                    if args.temp:
                        features = np.append(user_data[k][:][(-5+i)-j][1][:4], features)
                    else:
                        features = np.append(user_data[k][:][(-5+i)-j][1][:2], features)             
            
            #print(len(features))
            #print(len(usr_embeds[i][1]))
            usr_test_embeds = np.append(features, usr_embeds[i][1])
            test_data.append(usr_test_embeds)

            try:
                test_labels.append(usr_embeds[i+1][1][0]) # target is first element
            except:
                print('test labels out of bounds')
                print(uid)
                print(wid)
   
    #pprint(np.array(test_data))
    return test_data, test_labels

# Begin prepping data for training
print('Loading embeddings')
usr_embeddings = load_user_data()
#print('Loaded embeddings: ', usr_embeddings)

for k,v in usr_embeddings.iteritems():
    v.sort(key=lambda tup: tup[0]) # sort list of embed on week_id

num_users = len(usr_embeddings.keys())
total_embeddings = copy.deepcopy(usr_embeddings)

insample_mse = []
oos_mse = []
insample_corr = []
oos_corr = []

print('Total Folds: ', args.k_folds)

for i in range(args.k_folds):
    fold = i
    print('Starting Fold: ', fold)
    usr_embeddings = copy.deepcopy(total_embeddings) # reset to clean slate before folds
    
    holdout_data = {} # validation data
    # introducing a second test set, out of sample users
    # similar setup to holdout dev but is not used to tune alpha
    holdout_test_data = {}
    
    if args.k_folds > 1:
        lower_bound = fold*(num_users/args.k_folds)
        upper_bound = (fold+1)*(num_users/args.k_folds)
        print(lower_bound)
        print(upper_bound)
        for j in usr_embeddings.keys()[lower_bound:upper_bound]:
            try:
                holdout_data[j] = usr_embeddings[j]
            except:
                continue # no user wtih id = i
            del usr_embeddings[j]

            
        # last fold doesn't get evenly split cause above code removes
        # users that would be needed so we move the bounds manually
        if upper_bound > len(usr_embeddings.keys()):
            upper_bound = len(usr_embeddings.keys())
            lower_bound = upper_bound - len(holdout_data.keys())

        for j in usr_embeddings.keys()[lower_bound:upper_bound]:
            try:
                holdout_test_data[j] = usr_embeddings[j]
            except:
                continue
            del usr_embeddings[j]


    print('# keys usr_embed: ', len(usr_embeddings.keys()))
    print('# keys holdout: ', len(holdout_data.keys()))
    print('# keys oos test: ', len(holdout_test_data.keys()))

    #print('Generating train data')
    train_data, train_labels = gen_train_data_fixed(usr_embeddings, window)
    #print(train_data)
    #print(train_data.shape)
    #print(train_labels)
#print(train_data)
    #print('Generating test data')
    test_data, test_labels = gen_test_data(usr_embeddings, window)
    #print('Generating test 2 data')
    test2_data, test2_labels = gen_test_data(holdout_test_data, window)
    #print('Generating holdout data')
    holdout_test, holdout_labels = gen_test_data(holdout_data, window)


    train_df = pd.DataFrame(train_data)

    train_labels_df = pd.DataFrame(train_labels)
    train_labels_df.columns = ['label']
    train_labels_df['label'] = train_labels_df['label'].astype(float)

    test_df = pd.DataFrame(test_data).dropna()
    test2_df = pd.DataFrame(test2_data).dropna()

    holdout_df = pd.DataFrame(holdout_test).dropna()

    line_plot = my_plotter('line')
    alphas = [0,.001, .01, .1, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]
    mses = []
    coeffs = []
    inter = []
    models = []
    for a in alphas:
        if svr:
            model = SVR(kernel='linear')
        elif gbr:
            model = GradientBoostingRegressor(loss='lad')
        else:
            model = Ridge(alpha=a, normalize=False)
        
        model.fit(train_df, train_labels_df.label)
        h_preds = model.predict(holdout_df)
        mse = mean_squared_error(holdout_labels, h_preds)
        #pprint('holdout: ' + str(mse))
        mses.append(mse)
        
        models.append(model)
        
        if not svr and not gbr:
            coeffs.append(model.coef_)
            inter.append(model.intercept_)

        
    best_model = np.argmin(mses)
    best_alpha = alphas[best_model]
    print('Model info: ' )
    if not svr and not gbr:
        print(coeffs[best_model])
        print(inter[best_model])
        print('num coeff: ', len(coeffs[best_model]))

    print('best model: ', mses[best_model])
    print('best alpha: ', alphas[best_model])

    test_preds = models[best_model].predict(test_df)
    test2_preds = models[best_model].predict(test2_df)
    
    # don't test against missing data
    if args.daily_data:
        drop_indices = [i for i,x in enumerate(test_labels) if x == 0]
        test_labels = [test_labels[i] for i,x in enumerate(test_labels) if i not in drop_indices ]
        test_preds = [test_preds[i] for i,x in enumerate(test_preds) if i not in drop_indices ]
        
        drop_indices = [i for i,x in enumerate(test2_labels) if x == 0]
        test2_labels = [test2_labels[i] for i,x in enumerate(test2_labels) if i not in drop_indices ]
        test2_preds = [test2_preds[i] for i,x in enumerate(test2_preds) if i not in drop_indices ]

    test_mse = mean_squared_error(test_labels, test_preds)
    test2_mse = mean_squared_error(test2_labels, test2_preds)

    preds_labels = zip(test_preds, test_labels)


    save_obj(preds_labels, '/data/mmatero/mood_forecast/preds/LinAR_' + model_name)

    print('In Sample User Test MSE: ' + str(test_mse))
    print('OOS User Test MSE: ' + str(test2_mse))
    print('***********')
    
    insample_mse.append(test_mse)
    oos_mse.append(test2_mse)
    #print('Avg Cross Corr: ' + str(np.mean(corrs,axis=0)))

    #result_dict = dict()
    #undiff_result_dict = dict()

    #usr_diff_mse = [] # used for confidence intervals
    #labels = []
    #preds = []
    #for i,k in enumerate(usr_embeddings.keys()):
    #    dataset = '2k' if len(usr_embeddings.keys()) <= 2000 else '30k'
        
    #    usr_labels = test_labels[i*4:4*(i+1)]
    #    usr_preds = test_preds[i*4:4*(i+1)]
        
    #    labels.extend(usr_labels)
    #    preds.extend(usr_preds)
        #undiff_data = undiff_df[undiff_df['user_id'] == k]['affect_val'][-5:].tolist() # week 16-20
        #undiff_labels = undiff_df[undiff_df['user_id'] == k].affect_val.values.tolist()#undiff_data[-4:] # week 17-20
        #usr_undiff_preds = np.add(usr_preds,undiff_data[:4])

        #result_dict[k] = [usr_preds, usr_labels]
        #undiff_data = undiff_df[undiff_df['user_id'] == k]['affect_val'][-5:].tolist() # week 16-20
        #undiff_labels = undiff_df[undiff_df['user_id'] == k].affect_val.values.tolist()#undiff_data[-4:] # week 17-20
        #usr_undiff_preds = np.add(usr_preds,undiff_data[:4])
        #undiff_result_dict[k] = [usr_undiff_preds, undiff_labels]

    #    usr_diff_mse.append(mean_squared_error(usr_labels, usr_preds))


    #std_err = sem(usr_diff_mse)
    #print(std_err)
    #print(test_mse - (1.96*std_err))
    #print(test_mse + (1.96*std_err))
    #print(np.mean(usr_diff_mse)/2)
     
    print(len(usr_embeddings.keys()))
    test1_corr = pearsonr(test_labels, test_preds)
    test2_corr = pearsonr(test2_labels, test2_preds)
    insample_corr.append(test1_corr[0])
    oos_corr.append(test2_corr[0])
    print('In Sample Corr: ' + str(test1_corr))
    print('OOS Corr: ' + str(test2_corr))

print('################')
print('Avg insample MSE: ' + str(np.mean(insample_mse)))
print('Avg outsample MSE: ' + str(np.mean(oos_mse)))
print('Avg insample Corr: ' + str(np.mean(insample_corr)))
print('Avg outsample Corr: ' + str(np.mean(oos_corr)))
print('################')

#save_obj(result_dict, '/data/mmatero/user_results/diff_AR_' + str(window) + '_' + dataset)
#save_obj(undiff_result_dict, '/data/mmatero/user_results/undiff_AR_' + str(window) + '_' + dataset)
#save_obj(usr_diff_mse, '/data/mmatero/user_results/mse_AR_' + str(window) + '_' + dataset)
#file_name = str(k) + '-AR-undiff' + str(window)
            #line_plot.plot({'preds': usr_undiff_preds, 'labels': undiff_labels, }, file_name, dataset, action='save')

            #file_name = str(k) + '-AR-' + str(window)
            
            #line_plot.plot({'preds': usr_preds, 'labels': usr_labels, }, file_name, dataset, action='save')
