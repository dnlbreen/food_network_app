from flask import Flask,render_template,request,redirect
#from food_network_wrapper import recipe_search, get_n_recipes, scrape_recipe
#import matplotlib
#import seaborn as sns  # plots are prettier with Seaborn
#from wordcloud import WordCloud
#matplotlib.rcParams['savefig.dpi'] = 2 * matplotlib.rcParams['savefig.dpi']
#from matplotlib import rcParams
#rcParams.update({'figure.autolayout': True})

#import numpy as np
#import scipy as sp
#import pandas as pd
#import matplotlib.pylab as plt
#import json
#import os
#import time
#import re
#import gensim
#from gensim import corpora, models, similarities
#import pyLDAvis.gensim

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/four_topics', methods = ['GET', 'POST'])
def four_topics():
    return render_template('four_topics.html')

@app.route('/seven_topics', methods = ['GET', 'POST'])
def seven_topics():
    return render_template('seven_topics.html')
    
@app.route('/twenty_topics', methods = ['GET', 'POST'])
def twenty_topics():
    return render_template('twenty_topics.html')
    
if __name__ == '__main__':

    app.run()
