import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import time


class LGBM():


    def __init__(self, model_path, output_path, verbose=False):
        self.model_path = model_path
        self.output_path = output_path
        self.verbose = verbose
        if not os.path.exists(self.model_path): os.makedirs(self.model_path)
        if not os.path.exists(self.output_path): os.makedirs(self.output_path)


    def train(self, y, folds):
        K = len(folds)
        yhats = []
        rmses = [] 
        for k in range(K):
            start = time.time()

            np.random.seed(k)

            train = folds[k]['train']
            valid = folds[k]['valid']

            y_train, X_train = train[y], train.drop(columns=[y])
            y_valid, X_valid = valid[y], valid.drop(columns=[y])

            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

            params = {
                'boosting_type': 'gbdt',
                'objective': 'regression',
                #'metric': {'l2'},
                'metric': 'rmse',
                #'num_leaves': 51,
                #'max_depth': 5,
                'learning_rate': 0.01,
                #'feature_fraction': 0.7,
                #'bagging_fraction': 0.7,
                #'bagging_freq': 10,
                'verbose': -1,
                #'lambda_l1': 0.5,
                #'lambda_l2': 0.5,
                'first_metric_only': True,
                #'nthread': 3
            }

            params = {
                    'boosting_type': 'gbdt',
                    'objective': 'regression',
                    #'metric': {'l2'},
                    'metric': 'rmse',
                    'num_leaves': 4,
                    'max_bin': 200,
                    'learning_rate': 0.01,
                    'feature_fraction': 0.2,
                    'feature_fraction_seed': 7,
                    'bagging_fraction': 0.75,
                    'bagging_freq': 5,
                    'bagging_seed': 7,
                    'verbose': -1,
                    #'lambda_l1': 0.5,
                    #'lambda_l2': 0.5,
                    'first_metric_only': True,
                    #'nthread': 3
                }

            # train
            gbm = lgb.train(params,
                            lgb_train,
                            num_boost_round=5000,
                            valid_sets=[ lgb_eval ],
                            early_stopping_rounds=1000,
                            verbose_eval=False)

            y_pred = gbm.predict(X_valid, num_iteration=gbm.best_iteration)
            yhats.append(pd.DataFrame({ 'y': valid[y], 'yhat': y_pred }, index=valid.index))
            gbm.save_model(os.path.join(self.model_path, 'lgbm_{}.tar'.format(k)))
            rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
            rmses.append(rmse)

            if self.verbose:
                print('done training with {} fold and rmse={}. took {}s'.format(
                    k, round(rmse, 4), int(time.time() - start)))
        #y_test += np.exp(gbm.predict(test, num_iteration=gbm.best_iteration)) / K

        rmse = round(np.mean(rmses), 4)
        yhats = pd.concat(yhats).sort_index()
        yhats.to_csv(os.path.join(self.output_path, 'lgbm_{}'.format(rmse)))

        if self.verbose:
            print('avg rmse: {}'.format(rmse))
        
        return rmse


    def predict(self, test):
        model_files = [ os.path.join(self.model_path, f) for f in os.listdir(self.model_path) if f.startswith('lgbm_') ]
        K = len(model_files)
        y_test = np.zeros(test.shape[0])
        for m in model_files:
            gbm = lgb.Booster(model_file=m)
            y_test += gbm.predict(test, num_iteration=gbm.best_iteration) / K
        return y_test
        