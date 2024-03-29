"""
    This script aims at:
    - training a model using the data from a database (training set)
    - predicting labels on a testing set and reporting accuracy metrics
    - use GridSearch if required to look for the best combination of 
    hyperparameters
    - save the model's parameters as a pickle file that can be loaded later on

    It is assumed that this script will be ran from the highest level of
    directory (not from the "model" folder).

    Input:
    ------
    To launch the script, type the following command from the directory where
    the script is contained:
    python train_classifier.py DisasterResponse.db classifier.pkl conf.json

    where DisasterResponse.db is the database containing the data source
    classifier.pkl is the name to give to the pickle file created to save the
    model and conf.json contains the table to use in the database and all the
    optional parameters to train the model, such as whether to use grid search,
    with which hyperpameters...and whether we want to preprocess data for web
    app visualizations.

    Output:
    ------
    The pickle file saving the model will be created in the same directory
    where the script is contained.
    On the console will be printed:
    - the best parameters used by GridSearch (if -g was passed as an argument
    to the script)
    - the accuracy metrics of the prediction of the model

    Note on the parameters to use in the conf.json file
    - CountVectorizer is used, see sklearn doc to use the right hyperparameters
    - RandomForestClassifier is used, see sklearn doc to use the right
    hyperparameters
    - TF-IDF is currently used only with default hyperparameters, it is not
    possible to add some in the conf.file without updating the code of this
    script
    - if full_txt_process is true:
        - features__input_text_pipeline__vect__ to modify count_vectorizer
        parameters
        - features__input_text_pipeline__tfidf__ to modify tfidf parameters
        - clf__estimator__ to modify classifier parameters
    - if full_txt_process is false:
        - vect__ to modify count_vectorizer parameters
        - tfidf__ to modify tfidf parameters
        - clf__estimator__ to modify classifier parameters
    - if preprocess_viz is true, computing of data will be performed and
    dataframes will be saved to be used in the web app to build visualisations
"""

# import libraries
# note that sklearn's version should be at least 0.20.0

# import for NLP
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

# import for object manipulation
import argparse
import json
import numpy as np
import pandas as pd
import pickle
import os
import re
from sqlalchemy import create_engine

# import for ML pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from TextProcess import AverageWordLength, NumWords, NumStopWords, PopPerCat

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

parser = argparse.ArgumentParser()
parser.add_argument('db_path', help='Path or name of the database where the data is stored')
parser.add_argument('model_path', help='Path to the pickle file of the saved ML model')
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


def build_model(pipeline, grid_search, params):
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
        cv = GridSearchCV(pipeline, param_grid=params, cv=5)
        model = cv
        scenario = 'gridsearch_config'

    return model, scenario


def build_output_dict(output_dict, var_a, var_b):
    """
        Support function to add entries and values to a dictionary.
        This function is used to build the word_cat_dict, having either
        the word as a key or the category (var_a and var_b being either
        words of a message or list_of_cats).

        Inputs:
            output_dict - the dictionary to populate
            var_a - the set of values to use as keys
            var_b - the set of values to use as values and counts
        Outputs:
            The dictionary passed as an input is updated, no value
            is returned.
    """
    for a in var_a:
        if a not in output_dict:
            output_dict[a] = {}

        for b in var_b:
            if b not in output_dict[a]:
                output_dict[a][b] = 1
            else:
                output_dict[a][b] += 1


def comp_avg_length(X):
    """
        Compute the average length of a message.

        Inputs:
            X - the text to compute the average length for
        Outputs:
            X_avg_length as a dataframe of the average length
            of each document inside the text X.
    """
    X_avg_length = pd.Series(X).apply(word_length)
    return pd.DataFrame(X_avg_length)


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


def evaluate_model(grid_search, model, y_test, y_pred, scenario, print_report):
    """
        Evaluate the performance metrics of a model.

        Inputs:
            grid_search - whether grid_search was used to fit the model
            model - the model to use to evaluate its accuracy
            y_test - the reference labels for the training set
            y_pred - the predicting labels using the model
            scenario - the name of the scenario being evaluated (used
            in the metrics report)
            print_report - whether to print the metrics report
        Outputs:
            df_result - stored in memory
            printed report if print_report is true
    """
    print('Evaluating the model...')
    if grid_search:
        print('The best combination of parameters is the following: ', model.best_params_)

    results = compute_metrics(y_test, y_pred)
    df_results = pd.DataFrame()
    df_results = append_results_to_df(df_results, results, scenario)
    if print_report:
        print('Score metrics report:\n')
        print(df_results)


def fit_model(model, X_train, y_train):
    """
        Fit the model on X_train and y_train.
    """
    print('Fitting the model...')
    model.fit(X_train, y_train)


def get_list_of_cat(df):
    """
        Computes the list of categories associated to each row of df.

        Inputs:
            df - the dataframe containing the data available in the table
        Outputs:
            list_of_cats -> a dictionary, where the key is the row index
            and the value the list of categories associated to the row
    """
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

    list_of_cats = {}
    for row in range(len(df)):
        list_of_cat = []
        for k in range(0, df.shape[1]-4):
            if df.iloc[row, k] == 1:
                list_of_cat.append(cat_idx[k])
        list_of_cats[row] = list_of_cat

    return list_of_cats


def is_model(model_path):
    """
        Evaluate whether a pickle file with the model_path exists.
        If so, the saved model can be loaded and retrained.

        Inputs:
            model_path - the relative path to the saved model pickle file
        Outputs:
            bool - whether the path provided points to an existing file or not

    """
    if os.path.isfile(model_path):
        return True
    return False


def load_data(database_path, table_name):
    """
        Loads the data from the specified database using sqlachemy.

        Inputs:
            database - the database to connect to
            user_prompt/table_name - the name of the table to create or update
        Outputs:
            df -> a dataframe containing the data available in the table
    """

    print('Loading data...')
    engine = create_engine('sqlite:///{}'.format(database_path))
    if table_name:
        try:
            df = pd.read_sql('SELECT * FROM {}'.format(table_name), engine)
        except:
            raise
    else:
        raise AttributeError('Please provide the name of the table to use as a date source')

    return df


def load_model(model_path, grid_search, params):
    """
        If the model_path points to an existing pickle file, the model can be loaded
        as to be trained further using the parameters set in the conf.json file.

        Inputs:
            model_path - the path to the saved model pickle file
            grid_search - whether to use grid_search to train the model further
            params - the grid_search parameters to use
        Outputs:
            model - the instance of the model
            scenario - the name of the scenario to use to identify the model
            in the performance report
    """
    loaded_model = pickle.load(open(model_path, 'rb'))

    if 'cv' in loaded_model.get_params().keys():
        print('It seems like the saved model is a GridSearch instance.\n'
              'A new model will therefore be created using the hyperparameters set in the conf file.\n'
              'If you do not wish to continue, please stop the process and modify the conf file.')
        print('Best hyperparameters of the saved GridSearch model: ', loaded_model.best_params_)
        return (0, 0)

    else:
        if grid_search is False:
            model = loaded_model
            scenario = 'loaded_model'

        else:
            cv = GridSearchCV(loaded_model, param_grid=params, cv=5)
            model = cv
            scenario = 'loaded_gridsearch'

        return (model, scenario)


def predict_model(model, X_test):
    """
        Predict the labels for X_test using the model.
    """
    print('Predicting the categories...')
    y_pred = model.predict(X_test)
    return y_pred


def prepare_model(prepare_model_dict):
    """
        Processes information to know what model to use for the training.
        This function takes a dictionary as an input and depending on the
        values provided returns the appropriate combination of mode instance
        and scenario name.

        Inputs:
            prepare_model_dict - the dictionary of arguments
                prepare_model_dict = {'model_path': args.model_path,
                          'grid_search': grid_search,
                          'g_s_params': g_s_params, 'df': df,
                          'full_txt_process': full_txt_process,
                          'count_vect_params': count_vect_params,
                          'clf_params': clf_params,
                          'list_of_cats': list_of_cats}
        Outputs:
            model - the instance of the model
            scenario - the name of the scenario to use to identify the model
            in the performance report
    """

    model_path = prepare_model_dict['model_path']
    grid_search = prepare_model_dict['grid_search']
    g_s_params = prepare_model_dict['g_s_params']
    full_txt_process = prepare_model_dict['full_txt_process']
    df = prepare_model_dict['df']
    count_vect_params = prepare_model_dict['count_vect_params']
    clf_params = prepare_model_dict['clf_params']
    list_of_cats = prepare_model_dict['list_of_cats']

    print('Preparing the model...')
    use_loaded_model = is_model(model_path)
    if use_loaded_model is True:
        (model, scenario) = load_model(model_path, grid_search, g_s_params)
        if model == 0:
            use_loaded_model = False

    if use_loaded_model is False:
        if full_txt_process:
            word_cat_dict = word_count_per_cat(df, list_of_cats, 'full_txt_process')
            full_txt_params = [True, word_cat_dict]
            pipeline = structure_pipeline(df, count_vect_params, clf_params, full_txt_params)
        else:
            full_txt_params = [False, {}]
            pipeline = structure_pipeline(df, count_vect_params, clf_params, full_txt_params)

        model, scenario = build_model(pipeline, grid_search, g_s_params)

    return model, scenario


def preprocess_visualisation(list_of_cats, df):
    """
        Preprocessing of the data used to build visualiaions on the web app.

        Inputs:
            df - the dataframe containing the data available in the table
            list_of_cats - the list of categories associated with each row
            of the df
        Outputs:
            pickle files with the processed dataframes
    """

    print('Preprocessing the data for visualisations')
    # save preprocessed df for imbalance of class bar chart
    df_source = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    counts = []
    categories = list(df_source.columns.values)
    for i in categories:
        messages_tagged_1 = df_source[i].sum()
        messages_tagged_1 = messages_tagged_1*100/df_source.shape[0]
        messages_tagged_0 = df_source.shape[0]-df_source[i].sum()
        messages_tagged_0 = messages_tagged_0*100/df_source.shape[0]
        counts.append((i, messages_tagged_1, messages_tagged_0))
    data_stat = pd.DataFrame(counts, columns=['Disaster message category',
                                              'Distribution ratio - 1',
                                              'Distribution ratio - 0'])
    data_stat.sort_values(by=['Distribution ratio - 1'], inplace=True)
    pickle.dump(data_stat, open('data/df_class_imbalance.pkl', 'wb'))

    # save preprocessed df for word length scatter plot
    df_avg_length = pd.DataFrame()
    df_avg_length['index'] = df['id']
    df_avg_length['length'] = comp_avg_length(df['message'])['message']
    df_avg_length['genre'] = df['genre']
    data_length = df_avg_length[df_avg_length['length'] < 40]
    pickle.dump(data_length, open('data/df_word_length.pkl', 'wb'))

    # save proprocessed df for most popular word per category bar chart
    word_cat_dict = word_count_per_cat(df, list_of_cats, 'preprocess_viz')

    pop_word = pd.DataFrame()
    pop_word['category'] = word_cat_dict.keys()
    first_word = []
    first_word_count = []

    for category in word_cat_dict:
        sorted_array = sorted(word_cat_dict[category], key=word_cat_dict[category].get, reverse=True)[0:5]
        first_word.append(sorted_array[0])
        first_word_count.append(word_cat_dict[category][sorted_array[0]])

    pop_word['first_word'] = first_word
    pop_word['first_word_count'] = first_word_count
    pickle.dump(pop_word, open('data/df_pop_word.pkl', 'wb'))


def save_model(model_path, model):
    """
        Saves the model as a pickle file.

        Inputs:
            model_path - the path to the repository and name to give
            to the saved model
            model - the model instance to serialize
        Outputs:
            the model saved in a pickle file
    """
    print('Saving the model...')
    filename = model_path
    pickle.dump(model, open(filename, 'wb'))
    print('Model saved!')


def structure_pipeline(df, count_vect_params, clf_params, full_txt_params):
    """
        Constructs a pipeline with the appropriate estimators.

        Inputs:
            df - the dataframe containing the data available in the table
            count_vect_params - the hyperparameters of CountVectorizer
            clf_params - the hyperparameters of RandomForestClassifier
            full_txt_params - whether to use custom transformers and if so,
            the word_cat_dict
        Outputs:
            pipeline - the pipeline to use as a model
    """
    if full_txt_params[0]:
        word_cat_dict = full_txt_params[1]

        pipeline = Pipeline([
            ('features', FeatureUnion([
                ('input_text_pipeline', Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize, **count_vect_params)),
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
                    ('word_count_per_cat', PopPerCat(word_cat_dict)),
                ])),
            ])),

            ('clf', MultiOutputClassifier(RandomForestClassifier(**clf_params)))
        ])

    else:
        pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize, **count_vect_params)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier(**clf_params)))
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


def word_count_per_cat(df, list_of_cats, process):
    """
        Build a dictionary that matches categories with the count of words
        used in the messages labeled with these categories.
        The word_cat_dict dictionary is used both for web app visualisation
        and NLP feature engineering.

        Inputs:
            df - the dataframe containing the data available in the table
            list_of_cats - the list of categories associated with each row
            of the df
            process - whether we want categories as a key (process =
            'preprocess_viz') or the word as a key (process =
            full_txt_process')

        Outputs:
            word_cat_dict - a dictionary that matches categories with
            the count of words used in all messages labeled with these
            categories.
    """
    word_cat_dict = {}

    for row in range(len(df)):
        message = df.iloc[row, 1]
        words = tokenize(message)
        list_of_cat = list_of_cats[row]

        if list_of_cat != []:
            if process == 'preprocess_viz':
                build_output_dict(word_cat_dict, list_of_cat, words)
            elif process == 'full_txt_process':
                build_output_dict(word_cat_dict, words, list_of_cat)

    return word_cat_dict


def word_length(text):
    """
        Compute the average length of the words of a message.

        Inputs:
            text - the text to compute the average length for
        Outputs:
            avg_length - the average length of the words of
            the text
    """
    words = text.split()
    if len(words) > 0:
        avg_length = sum(len(word) for word in words)/len(words)
        return avg_length


def main():
    if args.model_path:
        if args.db_path:
            with open('model/conf.json', encoding='utf-8') as config_file:
                conf_data = json.load(config_file)

            data_table = conf_data['data_table']
            count_vect_params = conf_data['count_vect_params']
            clf_params = conf_data['clf_params']
            grid_search = conf_data['grid_search']
            g_s_params = conf_data['grid_search_parameters']
            full_txt_process = conf_data['full_txt_process']
            print_report = conf_data['print_report']
            preprocess_viz = conf_data['preprocess_viz']

            # we load the data from the database
            df = load_data(args.db_path, data_table)

            # define features and labels
            X = df['message']  # X is only the message column
            y = df.iloc[:, 4:]

            # we split the column 'message' of X and the whole y dataframe into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # we optionnaly compute the association row-list of categories for later use
            if full_txt_process or preprocess_viz:
                list_of_cats = get_list_of_cat(df)
            else:
                list_of_cats = {}

            # we build the pipeline
            prepare_model_dict = {'model_path': args.model_path,
                                  'grid_search': grid_search,
                                  'g_s_params': g_s_params, 'df': df,
                                  'full_txt_process': full_txt_process,
                                  'count_vect_params': count_vect_params,
                                  'clf_params': clf_params,
                                  'list_of_cats': list_of_cats}

            model, scenario = prepare_model(prepare_model_dict)

            # we fit the pipeline using the training sets
            fit_model(model, X_train, y_train)

            # we predict the categories using the testing set
            y_pred = predict_model(model, X_test)
            print(y_pred)

            # we compute the score metrics of the model's prediction
            evaluate_model(grid_search, model, y_test, y_pred, scenario, print_report)

            # we save the model
            save_model(args.model_path, model)

            # we preprocess the data for the visualisations
            if preprocess_viz:
                preprocess_visualisation(list_of_cats, df)

        else:
            print('Please specify the path or name of the target database where to load the data from')
    else:
        print('Please specify the path to the pickle file where the trained model was saved')


if __name__ == '__main__':
    main()
