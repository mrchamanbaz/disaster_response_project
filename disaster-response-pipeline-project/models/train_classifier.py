import sys
import pandas as pd
import sqlalchemy as db
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import GridSearchCV
from tensorflow import keras
from scikeras.wrappers import KerasClassifier
from sklearn.base import TransformerMixin, BaseEstimator
import pickle
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Tu run tensorflow on CPU



def load_data(database_filepath):
    # load data from database
    engine = db.create_engine(f'sqlite:///{os.path.abspath(database_filepath)}')
    print(engine.table_names())
    df = pd.read_sql_table('DisasterResponse.db', con=engine)
    X = df['message']
    column_names = df.columns
    Y = df[column_names[4:]]
    return X, Y, column_names


def tokenize(text):
    """
    :param text: the input text
    :return: the tokenized text
    This function cleans the tokenize the input text
    """
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # tokenize text
    tokens = word_tokenize(text)
    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens


def build_model():
    """
    :return: the machine learning model
    This function creates a classifier model. The parameters used here are the tuned ones which are defined
    by extensive simulation (in the notebook)
    """
    # here we insert the parameters leading to the best model
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(KNeighborsClassifier(n_jobs=-1)))
    ])

    parameters = {'vect__binary': [True],
                  'vect__max_df': [0.5],
                  'tfidf__smooth_idf': [True],
                  'tfidf__use_idf': [False],
                  'clf__estimator__leaf_size': [60],
                  'clf__estimator__n_neighbors': [20]
                  }
    f1_scorer = make_scorer(fbeta_score, beta=1, average='samples', zero_division=0)
    cv = GridSearchCV(pipeline, scoring=f1_scorer, param_grid=parameters, cv=2, verbose=3)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    :param model: the ML model
    :param X_test: Input test data
    :param Y_test: Input test labels
    :param category_names: name of categories
    :return:
    This function evaluates the trained model and prints out a classification report for
    all the category names.
    """
    y_pred = model.predict(X_test)
    for column_number, column_name in enumerate(Y_test.columns):
        print(f"Clasification Report for the feature {column_name} is:\n")
        print(classification_report(Y_test[column_name], y_pred[:, column_number], zero_division=0))


def save_model(model, model_filepath):
    """
    :param model: the tuned ML model
    :param model_filepath: filepath to save the model as pickle file
    :return:
    This function saves the trained ML model as a pickle file.
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()