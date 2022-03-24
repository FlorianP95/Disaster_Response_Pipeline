import json
import plotly
import pandas as pd
import re
import joblib

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine
from collections import Counter


app = Flask(__name__)


def tokenize(text, use_stemming=True):
    """
    Split text into words and return the stemmed&lemmatized or the lemmatized
    form of the words.

    Args:
        text (str): Disaster messages.
        use_stemming (bool, optional): Setting whether stemming is used or not.
            Defaults to True.

    Returns:
        clean_tokens (list of str): The input text broken into the original
            form of the words without stop words.
    """
    text = re.sub(r"[^\w\s]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        if tok in stopwords.words("english"):
            continue
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)

    if use_stemming:
        clean_tokens = [PorterStemmer().stem(w) for w in clean_tokens]
    return clean_tokens


# load data
engine = create_engine("sqlite:///..\\data\\DisasterResponse.db")
df = pd.read_sql_table("DisasterResponse", engine)

# load model
model = joblib.load("..\\models\\classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    labels = df.iloc[:, 4:]
    top_labels = labels.sum(axis=0).sort_values(ascending=False)[0:15]
    top_labels_names = list(top_labels.index)

    words = []
    for message in df["message"]:
        words.extend(tokenize(message)[:])

    words_counter = Counter(words)
    most_common = words_counter.most_common(7)
    most_common_words = []
    most_common_words_counts = []
    for word, count in most_common:
        most_common_words.append(word)
        most_common_words_counts.append(count)

    # create visuals
    graphs = [
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
                Bar(
                    x=top_labels_names,
                    y=top_labels
                )
            ],

            'layout': {
                'title': 'The 15 most common Labels',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Label"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=most_common_words,
                    y=most_common_words_counts
                )
            ],

            'layout': {
                'title': 'Occurrence of the most common words',
                'yaxis': {
                    'title': "Occurrence"
                },
                'xaxis': {
                    'title': "Top 7 words"
                }
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