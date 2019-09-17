from nltk.corpus import stopwords
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
nltk.download('stopwords')


class AverageWordLength(BaseEstimator, TransformerMixin):

    def word_length(self, text):
        words = text.split()
        if len(words) > 0:
            avg_length = sum(len(word) for word in words)/len(words)
            return avg_length
        return 0

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_avg_length = pd.Series(X).apply(self.word_length)
        return pd.DataFrame(X_avg_length)


class NumWords(BaseEstimator, TransformerMixin):

    def count_word(self, text):
        words = text.split()
        return len(words)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_num_words = pd.Series(X).apply(self.count_word)
        return pd.DataFrame(X_num_words)


class NumStopWords(BaseEstimator, TransformerMixin):

    def count_stop_word(self, text):
        words = text.split()
        c = 0
        for word in words:
            if words in stopwords.words('english'):
                c += 1
        return c

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_num_stop_words = pd.Series(X).apply(self.count_stop_word)
        return pd.DataFrame(X_num_stop_words)


class PopPerCat(BaseEstimator, TransformerMixin):

    def __init__(self, word_dict):
        self.word_cat_dict = word_dict

    def count_word_per_cat(self, text, word_cat_dict):
        words = text.split()
        count_per_cat = {}
        for word in words:
            if word in word_cat_dict:
                for cat in word_cat_dict[word]:
                    if cat not in count_per_cat:
                        count_per_cat[cat] = 1
                    else:
                        count_per_cat[cat] += 1
        return count_per_cat

    def fit(self, X, y=None):
        return self

    def get_params(self, deep=False):
        return {'word_dict': self.word_cat_dict}

    def transform(self, X):
        X_count_cat = pd.DataFrame()
        for x in X:
            x_count = self.count_word_per_cat(x, self.word_cat_dict)
            X_count_cat = X_count_cat.append(x_count, ignore_index=True)
        X_count_cat.fillna(0, inplace=True)
        return X_count_cat
