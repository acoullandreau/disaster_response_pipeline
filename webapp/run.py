from bokeh.palettes import viridis
import json
import numpy as np
import pandas as pd
import plotly
import re

from flask import Flask
from flask import render_template, request
import joblib
from sqlalchemy import create_engine

from wrangle_data import return_figures, tokenize

app = Flask(__name__)

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

    graphs = return_figures(df)

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
    print(classification_labels)
    print(classification_results)

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