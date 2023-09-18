import sys
# import libraries
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def load_data(database_filepath):
    '''
    load_data
    loads data from `database_filepath` into variables X, Y, category_names
    
    Input:
    database_filepath     filepath to disaster data
    
    Returns:
    X                      dataframe of target variable (message)
    Y                      dataframe of message categories
    category_names         list of category names
    '''
    
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterData', engine)
    X = df['message']
    y = df.iloc[:,4:]
    category_names = y.columns
    
    return X, y, category_names


def tokenize(text):
    '''
    tokenize
    tokenize message data
    
    Input:
    text       string of message data
    
    Returns:
    tokens     list of tokenized words
    '''
    
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
    '''
    build_model
    construct classification pipeline that uses gridsearch
    
    Input:
    
    Returns:
    pipeline    model pipeline
    '''
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier())),
    ])
    
    parameters = {'tfidf__use_idf':[True, False],
              'clf__estimator__n_estimators':[2,5],
              'clf__estimator__min_samples_leaf':[2,5]}

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    evaluate_model
    evaluates model based on classification and accuracy scores
    
    Input:
    model               model to be evalueated
    X_test              test messages
    Y_test              test message categories
    category_names      list of category names
    
    Returns:
    
    '''
    
    # predict on the test data
    y_pred = model.predict(X_test)
    for i, cat in enumerate(category_names):
        print(cat + '\n' + classification_report(Y_test.iloc[:,i], y_pred[:,i]) + '\n')


def save_model(model, model_filepath):
    '''
    save_model
    saves the model at the specified `model_filepath`
    
    Input:
    model              model to save
    model_filepath     filepath by which to save the model
    
    Returns:
                          
    '''
    
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