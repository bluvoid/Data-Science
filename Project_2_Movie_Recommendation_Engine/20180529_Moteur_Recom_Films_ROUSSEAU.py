# Moteur avec CAH N=50

from flask import Flask, request
import numpy as np
import pandas as pd
from sklearn import preprocessing, decomposition, cluster, metrics

# ouverture base num
data_films_clean_num_ = pd.read_csv(
    'DATA/data_films_clean_num.csv',    
    encoding='latin_1', 
    index_col=0,
    low_memory = False)

# "movie_imdb_link" n'est pas utilisé pour l'analyse, juste pour l'ID
data_films_clean_num = data_films_clean_num_.drop(['movie_imdb_link'], axis=1)

# ouverture base alpha
data_films_clean_alph = pd.read_csv(
    'DATA/data_films_clean_alph.csv',    
    encoding='latin_1', 
    index_col=0,
    low_memory = False)

# deconcatenation
genre_dumm = data_films_clean_alph['genres'].str.get_dummies(sep='|')
rating_dumm = data_films_clean_alph['content_rating'].str.get_dummies(sep='|')

# fusion des bases
data_films_final = pd.concat([data_films_clean_num, genre_dumm])
data_films_final = pd.concat([data_films_final, rating_dumm])
data_films_final = data_films_final.fillna(0)

# preprocessing
data_films_prepro = preprocessing.scale(data_films_final)

# cluster CAH N =50
cls_cah_50 = cluster.AgglomerativeClustering(n_clusters=50, 
                                affinity='euclidean', 
                                memory=None, 
                                connectivity=None, 
                                compute_full_tree='auto', 
                                linkage='ward') 
cls_cah_50.fit(data_films_prepro)

# fonction Moteur Recommandation
def recom(ID_film):
    ID_film = str(ID_film)
    ID_film_imdb_link = 'http://www.imdb.com/title/tt'+ID_film+'/?ref_=fn_tt_tt_1'

    base_cluster = pd.DataFrame(cls_cah_50.labels_, data_films_final.index.values, columns=['Cluster'])

    test = data_films_clean_alph[data_films_clean_alph['movie_imdb_link']==ID_film_imdb_link]
    movie_ind = test.index[0]
    #movie_ind =  movie_ind + '\xa0'
    movie_cluster = base_cluster.loc[movie_ind].values.item(0)
    films_cluster = base_cluster[base_cluster==movie_cluster]
    films_cluster = films_cluster.dropna(axis=0)
    recom = films_cluster.sample(5)

    _results=[]
    for i in recom.index:
        line={}
    
        movie_recom_name_ = i
        movie_recom_name_ = movie_recom_name_[:-1]
        line["Name"] = movie_recom_name_
    
        movie_recom_ID_ = data_films_clean_num_['movie_imdb_link'][i]
        movie_recom_ID_ = movie_recom_ID_[28:35]
        line["ID"] = movie_recom_ID_
      
        _results.append(line)
        
    return _results    


# Flask
app = Flask(__name__)

@app.route('/recommend/<int:ID_film>')
def test(ID_film):
    resultat = recom(ID_film)
    
    return str(resultat)

if __name__ == "__main__":
    app.run()