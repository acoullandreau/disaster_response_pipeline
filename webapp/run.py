from bokeh.palettes import viridis
import json
import numpy as np
import plotly
import pandas as pd
import re

from flask import Flask
from flask import render_template, request
from plotly.graph_objects import Bar, Heatmap, Scatter
import joblib
from sqlalchemy import create_engine

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')

app = Flask(__name__)


# tokenize function to process user input
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
    lemmatizer = WordNetLemmatizer()
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


# load data
database_path = 'data/DisasterResponse.db'
engine = create_engine('sqlite:///{}'.format(database_path))
table_name = 'MessagesWithCategory'
df = pd.read_sql_table(table_name, engine)

# load model
model_path = 'data/classifier.pkl'
model = joblib.load(model_path)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # support functions
    def build_color_palette(unique_target, target):
        color_palette = viridis(len(unique_target))
        color_item = {}
        i = 0
        for item_to_color in unique_target:
            color_select = color_palette[i].lstrip('#')
            color_select = tuple(int(color_select[i:i+2], 16) for i in (0, 2, 4))
            color_item[item_to_color] = color_select
            i += 1
        return color_item

    def get_color_scale(color_item, target):
        colors = ['rgb({0}, {1}, {2})'.format(color_item[item_to_color][0],
                                              color_item[item_to_color][1],
                                              color_item[item_to_color][2])
                  for item_to_color in target]
        return colors

    # visualize average length distribution
    def word_length(text):
        words = text.split()
        if len(words) > 0:
            avg_length = sum(len(word) for word in words)/len(words)
            return avg_length

    def comp_avg_length(X):
        X_avg_length = pd.Series(X).apply(word_length)
        return pd.DataFrame(X_avg_length)

    df_avg_length = pd.DataFrame()
    df_avg_length['index'] = df['id']
    df_avg_length['length'] = comp_avg_length(df['message'])['message']
    df_avg_length['genre'] = df['genre']
    data_length = df_avg_length[df_avg_length['length'] < 40]
    colors_avg_length = build_color_palette(df_avg_length['genre'].unique(), data_length['genre'])

    # visualize count per genre
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # visualize imbalance of classes
    df_source = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    counts = []
    categories = list(df_source.columns.values)
    for i in categories:
        messages_tagged_1 = df_source[i].sum()
        messages_tagged_1 = messages_tagged_1*100/df_source.shape[0]
        messages_tagged_0 = df_source.shape[0]-df_source[i].sum()
        messages_tagged_0 = messages_tagged_0*100/df_source.shape[0]
        counts.append((i, messages_tagged_1, messages_tagged_0))
    data_stat = pd.DataFrame(counts, columns=['Disaster message category', 'Distribution ratio - 1', 'Distribution ratio - 0'])
    data_stat.sort_values(by=['Distribution ratio - 1'], inplace=True)

    # visualize category correlation heat map
    data_corr = df.iloc[:, 4:]
    corr_list = []
    correl_val = data_corr.corr().values
    for row in correl_val:
        corr_list.append(list(row))

    graphs = [
        {
            'data': [
                Bar(
                    name='0',
                    x=data_stat['Disaster message category'],
                    y=data_stat['Distribution ratio - 0']),
                Bar(
                    name='1',
                    x=data_stat['Disaster message category'],
                    y=data_stat['Distribution ratio - 1'])
            ],
            'layout': {
                'barmode': 'group',
                'title': 'Distribution ratio of messages for each disaster category',
                'xaxis': {
                    'title': 'Category',
                    'tickangle': -45
                },
                'yaxis': {
                    'title': 'Percentage'
                },
                'margin': {
                    'l': 20,
                    'r': 20,
                    't': 20,
                    'b': 20
                }
            }
        },
        {
            'data': [
                Scatter(
                    name='direct',
                    x=data_length[data_length['genre'] == 'direct']['index'],
                    y=data_length[data_length['genre'] == 'direct']['length'],
                    showlegend=True,
                    mode='markers',
                    marker={
                        'color': colors_avg_length['direct']
                        }
                ),
                Scatter(
                    name='news',
                    x=data_length[data_length['genre'] == 'news']['index'],
                    y=data_length[data_length['genre'] == 'news']['length'],
                    showlegend=True,
                    mode='markers',
                    marker={
                        'color': colors_avg_length['news']
                        }
                ),
                Scatter(
                    name='social',
                    x=data_length[data_length['genre'] == 'social']['index'],
                    y=data_length[data_length['genre'] == 'social']['length'],
                    showlegend=True,
                    mode='markers',
                    marker={
                        'color': colors_avg_length['social']
                        }
                )
            ],

            'layout': {
                'title': "Average number of words per message",
                'labels': {"length": "Average length"}
            }
        },
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Heatmap(
                    z=corr_list,
                    x=data_corr.columns,
                    y=data_corr.columns,
                    colorscale='Viridis'
                )
            ],
            'layout': {
                'title': 'Correlation map of the categories',
                'height': 900
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()