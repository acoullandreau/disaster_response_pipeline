"""
    This script aims at:
    - training a model using the data from a database (training set)
    - predicting labels on a testing set and reporting accuracy metrics
    - use GridSearch if required to look for the best combination of hyperparameters
    - save the model's parameters as a pickle file that can be loaded later on

    Input:
    ------
    To launch the script, type the following command from the directory where the script is contained:
    python train_classifier.py DisasterResponse.db classifier.pkl [-g]

    where DisasterResponse.db is the database containing the data source
    classifier.pkl is the name to give to the pickle file created to save the model
    and -g is an optional parameter, if specified GridSearch will be executed

    Note that the table to use in the database will have to be specified after launching the script

    Output:
    ------
    The pickle file saving the model will be created in the same directory where the script is contained.
    On the console will be printed:
    - the best parameters used by GridSearch (if -g was passed as an argument to the script)
    - the accuracy metrics of the prediction of the model

"""

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
import pickle
import os

# import for ML pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from TextProcess import AverageWordLength, NumWords, NumStopWords, PopPerCat

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

parser = argparse.ArgumentParser()
parser.add_argument('db_path', help='Path or name of the database where to store the data')
parser.add_argument('model_path', help='Path to the pickle file of the saved ML model')
parser.add_argument('-g', '--grid_search', default=False, action="store_true", 
                    help='If true, gridsearch is executed to find best hyperparameters set')
parser.add_argument('-f', '--full_txt_process', default=False, action="store_true", 
                    help='If true, more complex pipeline is executed with additional text transformers')
args = parser.parse_args()

lemmatizer = WordNetLemmatizer()  # global variable used for the tokenization of the data


def append_results_to_df(df, results, scenario):
    """
        Appends the dataframe built with create_df_from_dict function to an existing
        results dataframe. This allows comparison of various scenarios.

        Inputs:
            df - the dataframe to append new results to
            results - the dictionary containing the metrics scores
            scenario - the tag name of the scenario associated with the series
            of metrics scores
        Outputs:
            df - a dataframe with each category as a row and each metric as a
            colum (precision, recall...) for all scenarios
    """

    if df.empty:
        df = create_df_from_dict(results, scenario)

    else:
        df_2 = create_df_from_dict(results, scenario)
        df = pd.concat([df, df_2], axis=1)

    df.sort_values('F1_score_{}'.format(scenario), inplace=True)

    return df


def build_model(pipeline, grid_search):
    """
        Define a pipeline with standard parameters and applies gridsearch with
        a list of predifined parameters if user launches the script with -g

        Inputs:
            grid_search - flag optionally passed as an argument when launching
            the script ; if set to True grid_search is applied
        Outputs:
            model - the model to be used to fit and predict
            scenario - tag used to label the results in the end
    """

    if grid_search is False:
        model = pipeline
        scenario = 'default_config'

    else:
        # let's set a list of parameters that have an influence on all estimators
        parameters = {
            'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
            'clf__estimator__n_estimators': [10, 100, 250, 500]
        }
        print("It's OK!")
        cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, cv=5)
        # we specify n_jobs = -1 as to be able to parallelise the execution of GridSearch

        print('The best combination of parameters is the following: ', cv.best_params_)
        model = cv
        scenario = 'gridsearch_config'

    return model, scenario


def compute_metrics(y_test, y_pred):
    """
        Generate classification_report (accuracy metrics agains y_test) for each categories of y_pred

        Inputs:
            y_test - the reference labels for the training set
            y_pred - the predicting labels using the model
        Outputs:
            reports - a dictionary with all the accuracy metrics of the comparison
            between y_test and y_pred
    """

    i = 0
    reports = {}
    for column in y_test.columns:
        report = classification_report(y_test[column], y_pred[:, i], labels=np.unique(y_pred[:, i]), output_dict=True)
        reports[column] = report['weighted avg']
        reports[column]['accuracy'] = (y_pred[:, i] == y_test[column]).mean()
        i += 1

    return reports


def create_df_from_dict(results, scenario):
    """
        Transforms the results dictionary into a dataframe.

        Inputs:
            results - the dictionary outputted by compute_metrics function
            scenario - the tag name of the scenario associated with the series of metrics scores
        Outputs:
            df - a dataframe with each category as a row and each metric as a colum (precision, recall...)
    """

    df = pd.DataFrame.from_dict(results, orient='index')
    df.drop('support', axis=1, inplace=True)
    df.rename(columns={"precision": 'Precision_{}'.format(scenario),
                       "recall": 'Recall_{}'.format(scenario),
                       "f1-score": 'F1_score_{}'.format(scenario),
                       "accuracy": 'Accuracy_{}'.format(scenario)}, inplace=True)
    return df


def is_model(model_path):
    if os.path.isfile(model_path):
        return True
    return False


def load_data(database_path):
    """
        Loads the data from the specified database using sqlachemy.

        Inputs:
            database - the database to connect to
            user_prompt/table_name - the name of the table to create or update
        Outputs:
            df -> a dataframe containing the data available in the table
    """

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


def load_model(model_path, grid_search):
    loaded_model = pickle.load(open(model_path, 'rb'))

    if grid_search is False:
        model = loaded_model
        scenario = 'loaded_model'

    else:
        parameters = input("Please provide the dictionary of hyperparameters to use with grid_search: ")

        cv = GridSearchCV(loaded_model, param_grid=parameters, n_jobs=-1, cv=5)
        # we specify n_jobs = -1 as to be able to parallelise the execution of GridSearch

        print('The best combination of parameters is the following: ', cv.best_params_)
        model = cv
        scenario = 'loaded_gridsearch'

    return model, scenario


def save_model(model_path, model):
    filename = model_path
    pickle.dump(model, open(filename, 'wb'))


def structure_pipeline(df, full_txt_process):

    if full_txt_process:
        print("It's OK!")
        word_cat_dict = word_count_per_cat(df)

        pipeline = Pipeline([
            ('features', FeatureUnion([
                ('input_text_pipeline', Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize, ngram_range=(1, 2))),
                    ('tfidf', TfidfTransformer()),
                ])),
                ('feat_eng_length', Pipeline([
                    ('average_length', AverageWordLength()),
                ])), 
                ('feat_eng_count', Pipeline([
                    ('word_count', NumWords()),
                ])), 
                ('feat_eng_stop_count', Pipeline([
                    ('stop_word_count', NumStopWords()),
                ])),
                ('feat_eng_pop', Pipeline([
                    ('word_count_per_cat', PopPerCat(word_dict=word_cat_dict)),
                ])),
            ])),

            ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1)))
        ])

    else:
        pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize, ngram_range=(1, 2),  max_df=0.5)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=250, random_state=42)))
        ])

    return pipeline


def tokenize(text):
    """
        Data cleaning and tranformation function, that parses the data and
        outputs a simplified data content.
        Parsing includes:
        - removal of the URLs
        - removal of the punctuation
        - tokenization of the text
        - lemmatization of the tokenized text (words)

        Inputs:
            text - the data to transform
        Outputs:
            words - the words contained in the data after parsing
    """

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


def word_count_per_cat(df):
    cat_idx = {0: 'related', 1: 'request', 2: 'offer', 3: 'aid_related',
               4: 'medical_help', 5: 'medical_products', 6: 'search_and_rescue',
               7: 'security', 8: 'military', 9: 'child_alone', 10: 'water',
               11: 'food', 12: 'shelter', 13: 'clothing', 14: 'money',
               15: 'missing_people', 16: 'refugees', 17: 'death',
               18: 'other_aid', 19: 'infrastructure_related', 20: 'transport',
               21: 'buildings', 22: 'electricity', 23: 'tools', 24: 'hospitals',
               25: 'shops', 26: 'aid_centers', 27: 'other_infrastructure',
               28:'weather_related', 29: 'floods', 30: 'storm', 31: 'fire',
               32: 'earthquake', 33: 'cold', 34: 'other_weather', 
               35: 'direct_report'}

    word_cat_dict = {}
    for row in range(len(df)):
        message = df.iloc[row, 1]
        list_of_cat = []
        for k in range(0, df.shape[1]-4):
            if df.iloc[row, k] == 1:
                list_of_cat.append(cat_idx[k])
        words = tokenize(message)

        if list_of_cat != []:
            for word in words:
                if word not in word_cat_dict:
                    word_cat_dict[word] = {}

                for cat in list_of_cat:
                    if cat not in word_cat_dict[word]:
                        word_cat_dict[word][cat] = 1
                    else:
                        word_cat_dict[word][cat] += 1
    return word_cat_dict


def main():
    if args.model_path:
        if args.db_path:
            # load data from database
            print('Loading data...')
            df = load_data(args.db_path)

            # define features and labels
            X = df['message']  # X is only the message column
            y = df.iloc[:, 5:]

            # we split the column 'message' of X and the whole y dataframe into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            print('Preparing the model...')
            if is_model(args.model_path):
                model, scenario = load_model(args.model_path, args.grid_search)

            else:
                if args.full_txt_process:
                    pipeline = structure_pipeline(df, True)
                else:
                    pipeline = structure_pipeline(df)

                model, scenario = build_model(pipeline, args.grid_search)

            print('Fitting the model...')
            # we fit the pipeline using the training sets
            model.fit(X_train, y_train)

            print('Predicting the categories...')
            # we predict the categories using the testing set
            y_pred = model.predict(X_test)

            print('Evaluating the model...')
            # we compute the score metrics of the model's prediction
            results = compute_metrics(y_test, y_pred)
            df_results = pd.DataFrame()
            df_results = append_results_to_df(df_results, results, scenario)
            print('Score metrics report:\n')
            print(df_results)

            # we save the model
            print('Saving the model...')
            save_model(args.model_path, model)
            print('Model saved!Ò')

        else:
            print('Please specify the path or name of the target database where to load the data from')
    else:
        print('Please specify the path to the pickle file where the trained model was saved')


if __name__ == '__main__':
    main()
