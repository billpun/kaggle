import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from sklearn.model_selection import KFold, StratifiedKFold
from typing import List, Dict
import pickle
import os
import pprint


class Encoder():


    def dummy(self, 
        data: DataFrame, columns:list=None, encoder:dict=None, 
        delimiter:str='.', remove_last:bool=True, keep_columns:bool=False) -> DataFrame:

        # if columns are not specified
        columns = columns if columns else list(data.columns)

        # if an encoder is not given
        if not encoder:
            # if not given and not defined
            encoder = {}
            for c in columns:
                if data[c].dtype == 'O':
                    encoder[c] = data[c].unique()
            print('A dummy encoder has been constructed.')

        dummy_cols = {}
        for c in columns:
            if c in self.encoder:
                values = self.encoder[c]
                for i, v in enumerate(values):
                    # remove last column if linearly dependent
                    if remove_last and i == len(values) - 1: 
                        continue
                    col = c + delimiter + str(v)
                    dummy_cols[col] = data[c].apply(lambda x : 1 if x == v else 0)

        #out = pd.concat([ data, pd.DataFrame(dummy_cols) ], axis=1)
        #if not keep_columns:
        #    out = out.drop(columns=self.encoder.keys())
        return {
            'encoded': encoder
            'columns': out
        }


    def save(self, filename: str):
        pickle.dump(self.encoder, open(filename, 'wb'))


    def load(self, filename: str):
        if not os.path.exists(filename):
            raise Exception('{} does not exists!'.format(filename))
        self.encoder = pickle.load(open(filename, 'rb'))

    
    def __test__():
        N = 10 
        de = DummyEncoder()
        print('********** Dummified Original Data **********')
        print('***** encoder      = new')
        print('***** remove_last  = False')
        print('***** keep_columns = True')
        print(de.run(pd.DataFrame({
            'X': np.random.randint(10, size=N),
            'class': np.random.choice(['A', 'B'], size=N)
        }), remove_last=False, keep_columns=True))
        print('********** Dummified New Data **********')
        print('***** encoder      = inherited (C and D ignored)')
        print('***** remove_last  = False')
        print('***** keep_columns = True')
        print(de.run(pd.DataFrame({
            'X': np.random.randint(10, size=N),
            'class': np.random.choice(['A', 'B', 'C', 'D'], size=N)
        }), remove_last=False, keep_columns=True))


class CvFold():

    
    def run_stratified(self, K, data, X, y, seed, shuffle):
        if data[y].dtype != 'int32' or data[y].dtype != 'int64':
            raise ValueError('Only support problems with integer targets!')
        tmp = data[X + [y]]
        f = StratifiedKFold(n_splits=K, random_state=seed, shuffle=shuffle)
        self.folds = []
        for _, j in f.split(tmp[X], tmp[[y]]):
            self.folds.append(list(j))


    def run(self, K, data, seed, shuffle):
        f = KFold(n_splits=K, random_state=seed, shuffle=shuffle)
        self.folds = []
        for _, j in f.split(data):
            self.folds.append(list(j))


    def __get_folds__(self, folds, level) -> Dict:
        out = []
        for i in range(len(folds)):
            _folds = folds[:i] + folds[i + 1:]
            tmp = {
                'train': sum(_folds, []),
                'valid': folds[i]
            }
            if level - 1 >= 0:
                lower_level = 'level_{}'.format(level - 1)
                tmp[lower_level] = self.__get_folds__(_folds, level - 1)
                tmp['train'] = []
                for k in tmp[lower_level]:
                    tmp['train'].append(k['valid'])
            out.append(tmp)
        return out

    
    def get_folds(self, level=1, with_data=True) -> Dict:
        return {
            'level_{}'.format(level - 1): 
            self.__get_folds__(self.folds, level - 1)
        }
        

    def get_nested_folds(self, level=1, with_data=True) -> List[List[Dict]]:
        out = []
        # loop thru the fold
        for k in range(self.K):
            lv1 = []
            # inner validation index
            folds = self.folds[:k] + self.folds[k+1:]
            for i in range(len(folds)):
                tmp = {
                    'train': sorted(sum(folds[:i] + folds[i + 1:], [])),
                    'valid': folds[i]
                }
                if with_data:
                    tmp['train'] = self.data.iloc[tmp['train']]
                    tmp['valid'] = self.data.iloc[tmp['valid']]
                lv1.append(tmp)

            tmp = {
                'lv1': lv1,
                'lv2': {
                    'train': sorted(np.hstack(folds)),
                    'valid': self.folds[k]
                }
            }
            if with_data:
                tmp['lv2']['train'] = []
                tmp['lv2']['valid'] = self.data.iloc[tmp['lv2']['valid']]
            out.append(tmp)
        return out


    def __test__():
        N = 24
        K = 4
        X = ['a', 'b']
        y = 'c'
        Y = 1
        data = pd.DataFrame(np.random.randint(N, size=(N, len(X))), columns=X)
        data[y] = np.random.randint(Y, size=N)
        cvf = CvFold()
        cvf.run(K, data, seed=0, shuffle=True)
        pp = pprint.PrettyPrinter(depth=10)
        pp.pprint(
            cvf.get_folds(level=3, with_data=False)
        )
        # print('********** Level 1 Only Cross-Validation Folds **********')
        # for d in cvf.get_folds():
        #     print('***** train *****')
        #     print(d['train'])
        #     print('***** valid *****')
        #     print(d['valid'])
        #     break

        # print('********** + Level 2 Cross-Validation Folds **********')
        # for d in cvf.get_nested_folds():
        #     for k, lv1 in enumerate(d['lv1']):
        #         print('***** k: {}: lv1 train *****'.format(k))
        #         print(lv1['train'])
        #         print('***** k: {}: lv1 valid *****'.format(k))
        #         print(lv1['valid'])
        #     print('***** lv2 valid *****')
        #     print(d['lv2']['train'])
        #     print(d['lv2']['valid'])
        #     break



if __name__ == "__main__":
    print('=============== Testing Cross-Validation Folds ===============')
    CvFold.__test__()
    # print('=============== Testing Dummy Encoder ===============')
    # DummyEncoder.__test__()