# import libraries
# note that sklearn's version should be at least 0.20.0

# import for NLP
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

# import for object manipulation
import argparse
import numpy as np
import pandas as pd
import re
from sqlalchemy import create_engine

# import for ML pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


parser = argparse.ArgumentParser()
parser.add_argument('db_path', help='Path or name of the database where to store the data')
parser.add_argument('model_path', help='Path to the pickle file of the saved ML model')
parser.add_argument('-g', '--grid-search', default=False, action="store_true", 
                    help='If true, gridsearch is executed to find best hyperparameters set')
args = parser.parse_args()

lemmatizer = WordNetLemmatizer()  # global variable used for the tokenization of the data


def load_data(database_path):
    engine = create_engine('sqlite:///{}'.format(database_path))
    table_name = input("Please provide the name of the table to use as a date source: ")
    if table_name:
        try:
            df = pd.read_sql('SELECT * FROM {}'.format(table_name), engine)
        except:
            raise
    else:
        table_name = input("Please provide the name of the table to use as a date source: ")

    return df


def get_feat_label(df, X_idx, y_idx):

    X = df.iloc[:, X_idx[0]:X_idx[1]]
    y = df.iloc[:, y_idx[0]:y_idx[1]]

    return X, y


def tokenize(text):
    # we convert the text to lower case
    text = text.lower()

    # we remove any url contained in the text

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    url_in_msg = re.findall(url_regex, text)
    for url in url_in_msg:
        text = text.replace(url, "urlplaceholder")

    # we remove the punctuation
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # we tokenize the text
    words = word_tokenize(text)

    # we lemmatize  and remove the stop words
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]

    return words


def compute_metrics(y_test, y_pred, global_avg=False):
    reports = {}
    if global_avg is False:
        i = 0
        for column in y_test.columns:
            report = classification_report(y_test[column], y_pred[:,i], labels=np.unique(y_pred[:,i]), output_dict=True)
            reports[column] = report
            i += 1

    report_global = classification_report(y_test, y_pred, labels=np.unique(y_pred[:,i]), output_dict=True)
    reports['global'] = report_global

    return reports


def print_report_metrics(reports):
    for report in reports:
        print('Classification report for "{}" category'.format(report))
        print('     F1 score (avg):', reports[report]['weighted avg']['f1-score'])
        print('     Precision (avg):', reports[report]['weighted avg']['precision'])
        print('     Recall (avg):', reports[report]['weighted avg']['recall'])
        print('\n')


def main():
    if args.model_path:
        if args.db_path:
            # load data from database
            df = load_data(args.db_path)

            # define features and labels
            X, y = get_feat_label(df, ['', 4], [5, ''])

            # we define the pipeline
            pipeline = Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
                ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=42), n_jobs=1))
            ])
            # for the MultiOutputClassifier, we initially set n_jobs to -1 but
            # this cause GridSearchCV to fail - so n_jobs is set back to 1.

            # we split the column 'message' of X and the whole y dataframe into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X['message'], y, test_size=0.2, random_state=42)

            if args.check_only is False:
                # we fit the pipeline using the training sets
                pipeline.fit(X_train, y_train)

                # we predict the categories using the testing set
                y_pred = pipeline.predict(X_test)

            else:
                # let's set a list of parameters that have an influence on all estimators
                parameters = {
                    'vect__ngram_range':[(1, 1), (1, 2)],
                    'vect__max_df':[0.5, 0.75, 1],
                    'vect__max_features':[None, 5000, 10000],
                    'clf__estimator__min_samples_leaf': [1, 2, 4],
                    'clf__estimator__min_samples_split': [2, 3, 4],
                    'clf__estimator__max_depth': [None, 10, 20, 50],
                    'clf__estimator__max_features': ['auto', 'log2'],
                    'clf__estimator__n_estimators':[10, 100, 250]
                }

                cv = GridSearchCV(pipeline, param_grid=parameters, cv=3) 
                # we specify cv=3, i.e the cross-validation splitting strategy - 3 folds 

                # let's repeat the fit and predict steps but now trying with all combinations of parameters set above
                cv.fit(X_train, y_train)
                y_pred = cv.predict(X_test)

            # we compute the score metrics of the model's prediction
            reports = compute_metrics(y_test, y_pred, global_avg=False)
            print_report_metrics(reports)

        else:
            print('Please specify the path or name of the target database where to load the data from')
    else:
        print('Please specify the path to the pickle file where the trained model was saved')


if __name__ == '__main__':
    main()
