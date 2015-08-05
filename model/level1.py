'''this file contains:

- split train data to l1-train and l2-train
- train level 1 base learners
    level 1 contains:
    - random forest
    - extra trees
    - xgb
    - xgb
    - knn
    - neural network
    - naive bayes
- produce train data pack for level 2
- produce test data pack for level 2

'''

import numpy as np
import pandas as pd
import theano
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum, adagrad
from nolearn.lasagne import NeuralNet

def train_data_from_level1(path, train_split):
    '''
    1. train level 1 base learners
    2. then load training data pack for level 2
    '''
    df = pd.read_csv(path)
    X = df.values.copy()
    # random shuffle the training data
    np.random.shuffle(X)
    X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]
    
    # label encoding
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels).astype(np.int32)

    # tf-idf transforming
    tfidf_trans = TfidfTransformer()
    X_tfidf = tfidf_trans.fit_transform(X).toarray().astype(np.float32)

    # 0-1 standardization
    standard_trans = StandardScaler()
    X_standard = standard_trans.fit_transform(X).astype(np.float32)
    
    # random forest on raw
    clf1 = RandomForestClassifier(n_estimators=300, n_jobs=-1)
    clf1.fit(X[:train_split], y[:train_split])
    pred1 = clf1.predict_proba(X[train_split:]).astype(np.float32)
    
    # extra trees on tfidf
    clf2 = ExtraTreesClassifier(n_estimators=300, n_jobs=-1)
    clf2.fit(X_tfidf[:train_split], y[:train_split])
    pred2 = clf2.predict_proba(X_tfidf[train_split:]).astype(np.float32)
    
    # xgb on raw
    dtrain3 = xgb.DMatrix(X[:int(train_split*0.8)], y[:int(train_split*0.8)])
    deval3 = xgb.DMatrix(X[int(train_split*0.8):train_split], y[int(train_split*0.8):train_split])
    dtest3 = xgb.DMatrix(X[train_split:])
    watchlist3 = [(dtrain3,'train'), (deval3,'eval')]
    param3 = {'max_depth':10, 'eta':0.0825, 'subsample':0.85, 'colsample_bytree':0.8, 'min_child_weight':5.2475 , 'objective':'multi:softprob', 'eval_metric':'mlogloss', 'num_class':9}
    num_rounds3 = 2000
    clf3 = xgb.train(param3, dtrain3, num_rounds3, watchlist3, early_stopping_rounds=15)
    pred3 = clf3.predict(dtest3, ntree_limit=clf3.best_iteration).astype(np.float32)
    
    # xgb on tfidf
    dtrain4 = xgb.DMatrix(X_tfidf[:int(train_split*0.8)], y[:int(train_split*0.8)])
    deval4 = xgb.DMatrix(X_tfidf[int(train_split*0.8):train_split], y[int(train_split*0.8):train_split])
    dtest4 = xgb.DMatrix(X_tfidf[train_split:])
    watchlist4 = [(dtrain4,'train'), (deval4,'eval')]
    param4 = {'max_depth':10, 'eta':0.0825, 'subsample':0.85, 'colsample_bytree':0.8, 'min_child_weight':5.2475 , 'objective':'multi:softprob', 'eval_metric':'mlogloss', 'num_class':9}
    num_rounds4 = 2000
    clf4 = xgb.train(param4, dtrain4, num_rounds4, watchlist4, early_stopping_rounds=15)
    pred4 = clf4.predict(dtest4, ntree_limit=clf4.best_iteration).astype(np.float32)
    
    # naive bayes on tfidf
    clf5 = MultinomialNB()
    clf5.fit(X_tfidf[:train_split], y[:train_split])
    pred5 = clf5.predict_proba(X_tfidf[train_split:]).astype(np.float32)
    
    # knn on tfidf with cosine
    clf6 = KNeighborsClassifier(n_neighbors=380, metric='cosine', algorithm='brute')
    clf6.fit(X_tfidf[:train_split], y[:train_split])
    pred6 = clf6.predict_proba(X[train_split:]).astype(np.float32)

    # nn on 0-1 standardized data
    num_classes = len(encoder.classes_)
    num_features = X_standard.shape[1]
    layers7 = [('input', InputLayer),
               ('dropoutf', DropoutLayer),
               ('dense0', DenseLayer),
               ('dropout', DropoutLayer),
               ('dense1', DenseLayer),
               ('dropout2', DropoutLayer),
               ('output', DenseLayer)]
    clf7 = NeuralNet(layers=layers7,
                     input_shape=(None, num_features),
                     dropoutf_p=0.15,
                     dense0_num_units=1000,
                     dropout_p=0.25,
                     dense1_num_units=500,
                     dropout2_p=0.25,
                     output_num_units=num_classes,
                     output_nonlinearity=softmax,

                     update=adagrad,
                     update_learning_rate=theano.shared(np.float32(0.01)),
                     max_epochs=50,
                     eval_size=0.2,
                     verbose=1,
                     )
    clf7.fit(X_standard[:train_split], y[:train_split])
    pred7 = clf7.predict_proba(X_standard[train_split:]).astype(np.float32)

    # combine raw with meta
    feat_pack = np.hstack((X[train_split:], pred1, pred2, pred3, pred4, pred5, pred6, pred7))
    return feat_pack, y[train_split:], tfidf_trans, standard_trans, clf1, clf2, clf3, clf4, clf5, clf6, clf7

def test_data_from_level1(path, tfidf_trans, standard_trans, clf1, clf2, clf3, clf4, clf5, clf6, clf7):
    '''
    1. load test data pack from level 1
    '''    
    df = pd.read_csv(path)
    X = df.values.copy()
    X= X[:, 1:].astype(np.float32)
    # transform to tfidf
    X_tfidf = tfidf_trans.transform(X).toarray().astype(np.float32)
    X_standard = standard_trans.transform(X).astype(np.float32)

    # pred proba with clf1 
    pred1 = clf1.predict_proba(X).astype(np.float32)
    # pred proba with clf2 
    pred2 = clf2.predict_proba(X_tfidf).astype(np.float32)
    # pred proba with clf3
    pred3 = clf3.predict(xgb.DMatrix(X), ntree_limit=clf3.best_iteration).astype(np.float32)
    # pred proba with clf4 
    pred4 = clf4.predict(xgb.DMatrix(X_tfidf), ntree_limit=clf4.best_iteration).astype(np.float32)
    # pred proba with clf5
    pred5 = clf5.predict_proba(X_tfidf).astype(np.float32)
    # pred proba with clf6
    pred6 = clf6.predict_proba(X_tfidf).astype(np.float32)
    # pred proba with clf7
    pred7 = clf7.predict_proba(X_standard).astype(np.float32)

    # combine raw with meta
    feat_pack = np.hstack((X, pred1, pred2, pred3, pred4, pred5, pred6, pred7))
    return feat_pack