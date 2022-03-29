import sys
import pickle
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

from sqlalchemy import create_engine

import pandas as pd
import nltk

nltk.download(['punkt', 'wordnet', 'stopwords'])


def load_data(database_filepath):
    """
    Load the daa from a SQL database.

    Args:
        database_filepath (str): Filepath to the db.

    Returns:
        X (Pandas DataFrame): Dataframe with the features (messages).
        y (Pandas DataFrame): Dataframe with the labels (categories).
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql("DisasterResponse.db", engine)
    X = df["message"]
    y = df.iloc[:, 4:]
    return X, y


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


def build_model():
    """
    Build a pipeline for NLP-Multi-Output-Classification.

    Returns:
        (sklearn Pipeline): Pipeline for NLP Classification tasks.
    """
    classifier = RandomForestClassifier()
    pipeline = Pipeline([
        ("vect", CountVectorizer(tokenizer=tokenize)),
        ("tfidf", TfidfTransformer()),
        ("model", MultiOutputClassifier(classifier))
    ])

    parameters = {
        'tfidf__use_idf': (True, False),
        'model__estimator__n_estimators': [50, 100],
    }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test,):
    """
    Test the model with the test-dataset and print the f1 score,
    precision and recall for each output category of the dataset.

    Args:
        model (sklearn Pipeline): The NLP classification Pipeline.
        X_test (Pandas DataFrame): DF containing the test messages.
        y_test (Pandas DataFrame): DF containing the test labels.
    """
    Y_pred = model.predict(X_test)
    # Printing the classification report for each label
    for count, column in enumerate(Y_test.columns):
        print("Label:", column)
        print(classification_report(Y_test[column], Y_pred[:, count]))

    accuracy = (Y_pred == Y_test.values).mean()
    print(f"model accuracy {accuracy}")


def save_model(model, model_filepath):
    """
    Saves a pickle file of the model.

    Args:
        model (sklearn Pipeline): The NLP classification Pipeline.
        model_filepath (str): Path of the pickel file.
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print(f"Loading data...\n    DATABASE: {database_filepath}")
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                            test_size=0.2)

        print("Building model...")
        model = build_model()

        print("Training model...")
        model.fit(X_train, Y_train)

        print("Evaluating model...")
        evaluate_model(model, X_test, Y_test)

        print(f"Saving model...\n    MODEL: {model_filepath}")
        save_model(model, model_filepath)

        print("Trained model saved!")

    else:
        print("Please provide the filepath of the disaster messages database "
              "as the first argument and the filepath of the pickle file to "
              "save the model to as the second argument. \n\nExample: python "
              "train_classifier.py ../data/DisasterResponse.db classifier.pkl")


if __name__ == "__main__":
    main()
