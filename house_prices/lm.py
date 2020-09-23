import os
import numpy as np
import pandas as pd
import time
import pickle
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


class LM():


    def __init__(self, model_path, output_path, model, verbose=False):
        self.model_path = model_path
        self.output_path = output_path
        self.model = model # en or lm
        self.verbose = verbose
        if self.model not in ['en', 'lm']:
            raise ValueError('Model {} not supported!'.format(self.model))
        if not os.path.exists(self.model_path): os.makedirs(self.model_path)
        if not os.path.exists(self.output_path): os.makedirs(self.output_path)


    def train(self, y, folds, l1_ratio=0.8, normalize=True):
        K = len(folds)
        yhats = []
        rmses = [] 
        for k in range(K):
            start = time.time()
            
            train = folds[k]['train']
            valid = folds[k]['valid']
            
            X_train = train.drop(columns=[y]).values
            y_train = train[y].values
            X_valid = valid.drop(columns=[y]).values
            y_valid = valid[y].values
            
            if self.model == 'en':
                kf = KFold(n_splits=10, random_state=100 + k, shuffle=True)
                lm = ElasticNetCV(cv=kf, random_state=k, normalize=normalize, max_iter=5000, l1_ratio=l1_ratio)
            elif self.model == 'lm':
                lm = LinearRegression(normalize=normalize)
            
            lm.fit(X_train, y_train)
            y_pred = lm.predict(X_valid)
            yhats.append(pd.DataFrame({ 'y': valid[y], 'yhat': y_pred }, index=valid.index))
            pickle.dump(lm, open(os.path.join(self.model_path, '{}_{}.tar'.format(self.model, k)), 'wb'))
            rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
            rmses.append(rmse)
            
            if self.verbose:
                print('done training with {} fold and rmse={}. took {}s'.format(
                    k, round(rmse, 4), int(time.time() - start)))
            
        rmse = round(np.mean(rmses), 4)
        yhats = pd.concat(yhats).sort_index()
        yhats.to_csv(os.path.join(self.output_path, '{}_{}'.format(self.model, rmse)))
        
        if self.verbose:
            print('avg rmse: {}'.format(rmse))

        return rmse

    
    def predict(self, test):
        model_files = [ 
            os.path.join(self.model_path, f) for f in os.listdir(self.model_path) \
                if f.startswith('{}_'.format(self.model)) ]
        K = len(model_files)
        y_test = np.zeros(test.shape[0])
        for m in model_files:
            lm = pickle.load(open(m, 'rb'))
            y_pred = lm.predict(test)
            y_test += y_pred / K
        return y_test