import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """

        """
        BIC Equation:  BIC = -2 * logL + p * logN
          L -> likelihood of fitted model
          p -> no of parameters
          N -> no of data points (dataset size)
          p * log N -> penalty term to penalize complexity
          Ref: http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
          https://discussions.udacity.com/t/parameter-in-bic-selector/394318/8

        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        min_bic = float('inf')
        best_model = None

        for state in range(self.min_n_components, self.max_n_components + 1):
            try:
                hmm_model = self.base_model(state)
                logL = hmm_model.score(self.X, self.lengths)
                logN = math.log(len(self.X))
                p = state**2 + 2*state* hmm_model.n_features - 1
                bic = -2*logL + p*logN
                if bic < min_bic:
                    min_bic = bic
                    best_model = hmm_model
            except:
                continue

        if best_model is None:
            return self.base_model(self.n_constant)

        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score = float('-inf')
        best_model = None

        for state in range(self.min_n_components, self.max_n_components + 1):
            try:
                hmm_model = self.base_model(state)
                org_score = hmm_model.score(self.X, self.lengths)

                others_score_sum = 0.0
                other_words_count = 0
                # calculate anti log likelihoods
                for word in self.words:
                    if word == self.this_word:
                        continue

                    x_other, lengths_other = self.hwords[word]
                    logL = hmm_model.score(x_other, lengths_other)
                    others_score_sum += logL
                    other_words_count += 1

                others_score_avg = others_score_sum/other_words_count
                curr_DIC = org_score - others_score_avg

                if curr_DIC>best_score:
                    best_score = curr_DIC
                    best_model = hmm_model

            except:
                pass

        if best_model is None:
            return self.base_model(self.n_constant)

        return best_model




class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds
        One technique for cross-validation is to break the training set into "folds" and rotate which fold is left out
        #  of training. The "left out" fold scored
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score = float('-inf')
        best_model = None

        #Split the training set into folds for cross-validation and score accordingly
        if len(self.sequences) > 1:
            split_method = KFold(n_splits=min(3, len(self.sequences)))

            for state in range(self.min_n_components, self.max_n_components + 1):
                count = 0
                sum_LogL = 0

                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    x_train, lengths_train = combine_sequences(cv_train_idx, self.sequences)
                    x_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)

                    try:
                        hmm_model = self.base_model(state)
                        logL = hmm_model.score(x_test, lengths_test)
                        count += 1

                    except:
                        logL = 0

                    sum_LogL += logL
                # calculate average score
                cv_score = sum_LogL/ (1 if count == 0 else count)

                # find best model based on score
                if cv_score > best_score:
                    best_score = cv_score
                    best_model = hmm_model

        if best_model is None:
            return self.base_model(self.n_constant)

        return best_model

