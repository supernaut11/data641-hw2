from collections import Counter
from imblearn.over_sampling import RandomOverSampler, SMOTEN
import llr
import numpy as np

class Oversampler:
    def __init__(self, algorithm):
        self._alg = algorithm

    def fit_resample(self, X_train, y_train):
        X_over, y_over = self._alg.fit_resample(np.array(X_train, dtype=object).reshape([-1, 1]), 
                                                np.array(y_train, dtype=object).reshape([-1, 1]))

        # X_over is actually an array of arrays, eliminate first level of indirection before returning
        X_over = [x[0] for x in X_over]
        return X_over, y_over

class RandomOversampler(Oversampler):
    def __init__(self, random_state=42):
        super().__init__(RandomOverSampler(random_state=random_state))

class SmoteOversampler(Oversampler):
    def __init__(self, random_state=42):
        super().__init__(SMOTEN(random_state=random_state))

class LlrReduction:
    def __init__(self, X_train, y_train, X_test):
        self._X_train = X_train
        self._y_train = y_train
        self._X_test = X_test

    def _reduce_helper(self, X_samples, top_x1, top_x2, min_x):
        X_new = []
        for x in X_samples:
            cur_str = ' '.join([w for w in x.split() if w in top_x1 or w in top_x2])
            if len(cur_str.split()) < min_x:
                cur_str = x
            
            X_new.append(cur_str)
        
        return X_new

    def reduce_features(self, llr_factor, label1, label2, min_x=0):
        x1 = [self._X_train[i] for i in range(0, len(self._X_train)) if self._y_train[i] == label1]
        x2 = [self._X_train[i] for i in range(0, len(self._X_train)) if self._y_train[i] == label2]
        x1_counter = Counter(' '.join(x1).split())
        x2_counter = Counter(' '.join(x2).split())

        cmp_results = llr.llr_compare(x1_counter, x2_counter)

        top_x1 = {k:v for k,v in sorted(cmp_results.items(), key=lambda x: (-x[1], x[0]))[:llr_factor]}
        top_x2 = {k:v for k,v in sorted(cmp_results.items(), key=lambda x: (x[1], x[0]))[:llr_factor]}

        X_train_new = self._reduce_helper(self._X_train, top_x1, top_x2, min_x)
        X_test_new = self._reduce_helper(self._X_test, top_x1, top_x2, min_x)

        return X_train_new, X_test_new