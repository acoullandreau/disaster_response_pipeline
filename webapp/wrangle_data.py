from bokeh.palettes import viridis
import joblib
import pandas as pd
from plotly.graph_objects import Bar, Heatmap, Scatter
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')


# support functions
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


# build visualisation graphs
def return_figures(df):
    figures = []

    # visualize imbalance of classes
    graph_one = []
    data_stat = joblib.load('data/df_class_imbalance.pkl')

    graph_one.append(
        Bar(
            name='0',
            x=data_stat['Disaster message category'],
            y=data_stat['Distribution ratio - 0']),
        Bar(
            name='1',
            x=data_stat['Disaster message category'],
            y=data_stat['Distribution ratio - 1'])

    )

    layout_one = dict(barmode='group',
                      title='Distribution ratio of messages for each disaster category',
                      yaxis=dict(title="Percentage"),
                      xaxis=dict(title="Category", tickangle=-45))

    # visualize average message length
    graph_two = []
    data_length = joblib.load('data/df_word_length.pkl')

    graph_two.append(
        Scatter(
            name='direct',
            x=data_length[data_length['genre'] == 'direct']['index'],
            y=data_length[data_length['genre'] == 'direct']['length'],
            showlegend=True,
            mode='markers',
        ),
        Scatter(
            name='news',
            x=data_length[data_length['genre'] == 'news']['index'],
            y=data_length[data_length['genre'] == 'news']['length'],
            showlegend=True,
            mode='markers',
        ),
        Scatter(
            name='social',
            x=data_length[data_length['genre'] == 'social']['index'],
            y=data_length[data_length['genre'] == 'social']['length'],
            showlegend=True,
            mode='markers',
        )
    )

    layout_two = dict(title='Average number of words per message',
                      labels=dict(length="Average length"))

    # visualize count per genre
    graph_three = []
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    graph_three.append(
        Bar(
            x=genre_names,
            y=genre_counts,
        ),

    )

    layout_three = dict(title='Distribution of Message Genres',
                        yaxis=dict(title="Count"), xaxis=dict(title="Genre"))

    # visualize category correlation heat map
    graph_four = []
    data_corr = df.iloc[:, 4:]
    corr_list = []
    correl_val = data_corr.corr().values
    for row in correl_val:
        corr_list.append(list(row))

    graph_four.append(
        Heatmap(
            z=corr_list,
            x=data_corr.columns,
            y=data_corr.columns,
            colorscale='Viridis'
        )

    )

    layout_four = dict(title='Correlation map of the categories',
                       height=900)

    # visualize most frequent word per category

    figures.append(dict(data=graph_one, layout=layout_one))
    figures.append(dict(data=graph_two, layout=layout_two))
    figures.append(dict(data=graph_three, layout=layout_three))
    figures.append(dict(data=graph_four, layout=layout_four))

    return figures
