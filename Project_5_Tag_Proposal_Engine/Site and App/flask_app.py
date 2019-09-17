# 20180829
# Tag proposal engine ("TFIDF + Clustering + Random Forest" or POS Tagging)

from flask import Flask, request, render_template
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing, linear_model
from sklearn.ensemble import RandomForestClassifier

import nltk
from nltk.tokenize import word_tokenize
nltk.download("stopwords")
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


# Flask
app = Flask(__name__)

words_ok = []

# DATABASE OPENING (already cleaned: beautifulSoup, Duplicate removal, Lemmatization) --- -
data_text = pd.read_csv(
    '/home/mysite/DATA_clean_P5/20180925_sansdouble_data_OK_10000_Clus_tfidf_LDA_OK.csv',
    index_col=0,
    encoding='utf-8',
    low_memory = False)


# TFIDF sur X  --------

# 1) X TFIDF
tfidf = TfidfVectorizer(max_features = 400, 
                        max_df = 1300,
                        min_df = 150)
values_X_tfidf = tfidf.fit_transform(data_text['Text_OK'])
X_tfidf = pd.DataFrame(values_X_tfidf.toarray(), columns = [tfidf.get_feature_names()])

scaler_tags_x_tfidf = preprocessing.StandardScaler().fit(X_tfidf)
X_tfidf_scaled = scaler_tags_x_tfidf.transform(X_tfidf)


# PREDICTION ------

# POS Tag: NN = singular noun & NNS =  plurial noun
def pos(x):
    u = word_tokenize(x)
    kkk = nltk.pos_tag(u)
    tt = [w for w,x in kkk if x =='NN' or x =='NNS']
    ttt = " ".join(tt)
    tttt = unique(ttt.split())
    if not tttt:
        tttt = 'Not enough words for Tagging !'
    return tttt


def preditag(words_ok):
    # FIT TFIDF with tfidf already done on train set
    question_tfidf_ = tfidf.transform(words_ok)
    question_f = pd.DataFrame(question_tfidf_.toarray(), columns = [tfidf.get_feature_names()])
         
    # Random Forest to predict y Cluster   
    rfcCV = RandomForestClassifier(n_estimators=4,
                                   max_depth=5)
    rfcCV.fit(X_tfidf_scaled, data_text['y Freq Cluster'])
    
    # PREDICT
    # We search the cluster it belongs
    Cluster_tags_predit  = rfcCV.predict(question_f)
    Cluster_tags_predit  = Cluster_tags_predit[0]

    tags_cluster = data_text[data_text['y Freq Cluster'] == Cluster_tags_predit]
    tags_cluster = tags_cluster.dropna(axis=0)
    
    # !!! we take the most frequent tag in the predicted cluster
    recom = tags_cluster.sample(frac = 0.05)  
      
    #recom = tags_cluster
    tags_unprocessed = " ".join(recom['Tags_OK_Freq'].values)
    tags_recom = unique(tags_unprocessed.split())
      
    return tags_recom


def unique(l):
    ulist = []
    [ulist.append(x) for x in l if x not in ulist]
    koo = ' / '.join(ulist)
    return koo


# Engine: Tag Generator  ---------------------------------------------------------------------------
@app.route('/', methods=['GET', 'POST'])
def index(result = None):
    if request.method == 'POST':
        if request.form['question']:
            questio = request.form['question']
            questio_ti = request.form['question_title']
            question_merge = questio + " " + questio_ti
                      
            #values from X tfidf :
            voca = tfidf.vocabulary_
            
            # extract word in "voca"
            words_ok = [x for x in voca.keys() if x in question_merge]
            if not words_ok:
                result = pos(question_merge)
                result = str(result)
                result = "POS TAGGING TAGS GENERATION: " + result 
            else:
                result = preditag(words_ok)
                result = str(result)
                result = "TAGS PREDICTED with X_tfidf/CAH y_Freq: " + result
                
    return render_template('main_page.html', result = result)