'''this file contains:

- data loader
- data submit

'''
import numpy as np
import pandas as pd

def load_data(train_path, test_path, train_split):
    train_x, train_y, tfidf_trans, standard_trans, clf1, clf2, clf3, clf4, clf5, clf6, clf7 = train_data_from_level1(train_path, train_split)
    test_x = test_data_from_level1(test_path, tfidf_trans, standard_trans, clf1, clf2, clf3, clf4, clf5, clf6, clf7)
    return train_x, train_y, test_x

def submit(submit_arr):
    sample_submit_df = pd.read_csv(r'./data/sampleSubmission.csv')
    submit_arr = np.insert(submit_arr, 0, range(1, sample_submit_df.shape[0]+1), axis=1)
    submit_df = pd.DataFrame(submit_arr, columns=sample_submit_df.columns)
    submit_df['id'] = submit_df['id'].apply(lambda x: int(x))
    submit_df.to_csv('submission.csv', index=False)