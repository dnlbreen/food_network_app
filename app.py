from flask import Flask,render_template,request,redirect,jsonify
#from food_network_wrapper import recipe_search, get_n_recipes, scrape_recipe
#import matplotlib
#import seaborn as sns  # plots are prettier with Seaborn
#from wordcloud import WordCloud
#matplotlib.rcParams['savefig.dpi'] = 2 * matplotlib.rcParams['savefig.dpi']
#from matplotlib import rcParams
#rcParams.update({'figure.autolayout': True})

import numpy as np
import scipy as sp
import cPickle
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
import difflib
#import pandas as pd
#import matplotlib.pylab as plt
#import json
#import os
#import time
#import re
#import gensim
#from gensim import corpora, models, similarities
#import pyLDAvis.gensim

def train_svm(X, y):
    """
    Create and train the Support Vector Machine.
    """
    #maybe use logistic regression and linear svc if sample size grows
    svm = SVC(C=1000000.0, gamma=0.0, kernel='rbf')
    svm.fit(X, y)
    return svm

def create_tfidf_training_data(features):

    # Create the TF-IDF vectoriser and transform the corpus
    vectorizer = TfidfVectorizer(min_df=1)
    X = vectorizer.fit_transform(features)
    return X

def get_category(model, query, vectorizer, dense_zeros):

    dense_vector = dense_zeros
    
    features = [difflib.get_close_matches(ingredient, vectorizer.vocabulary_.keys(), n=1)[0] for ingredient in query.split(' ')]
    
    indices = [vectorizer.vocabulary_[feature] for feature in features]
    
    for index in indices:
    
        dense_vector[index] = vectorizer.idf_[index]

    query = sp.sparse.csr_matrix(dense_vector)
    
    print model.predict(query)[0]
    
    return model.predict(query)[0]

#Load in the svm

with open('models/svm_classifier.pkl', 'rb') as fid:
    svm_loaded = cPickle.load(fid)

#load in the corpus and create reference tfidf vectorizer

corpus_even = []

file = open('models/corpus_even.txt')

for n, line in enumerate(file):

    corpus_even.append(line.rstrip('\n'))

vectorizer = TfidfVectorizer(min_df=1)
_ = vectorizer.fit_transform(corpus_even)

dense_zeros = np.zeros(len(vectorizer.idf_))

X = create_tfidf_training_data(corpus_even)

labels = np.loadtxt('models/labels.txt', dtype = str)

svm_loaded = train_svm(X, labels)

app = Flask(__name__)

#app.vars = {}

@app.route('/')
def index():
    #if request.method == 'GET':
    return render_template('index.html')
    #else:
        #app.vars['ingredients'] = request.form['ingredients']
        
        #query = app.vars['ingredients']
        
        #return render_template('response.html',category=get_category(svm_loaded, query, vectorizer, dense_zeros))

@app.route('/four_topics', methods = ['GET', 'POST'])
def four_topics():
    return render_template('four_topics.html')

@app.route('/seven_topics', methods = ['GET', 'POST'])
def seven_topics():
    return render_template('seven_topics.html')
    
@app.route('/twenty_topics', methods = ['GET', 'POST'])
def twenty_topics():
    return render_template('twenty_topics.html')

@app.route('/query')
def query():

    ing = request.args.get('ing', 'nothing', type=str)
    print 'got here'
    return jsonify(result=get_category(svm_loaded, ing, vectorizer, dense_zeros))
    
@app.route('/_add_numbers')
def add_numbers():
    a = request.args.get('a', 0, type=str)
    b = request.args.get('b', 0, type=str)
    return jsonify(result=get_category(svm_loaded, a + b, vectorizer, dense_zeros))
if __name__ == '__main__':

    app.run()
